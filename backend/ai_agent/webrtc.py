import logging
import uuid
import asyncio
import av
import fractions
import time
import os
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from channels.db import database_sync_to_async
from aioice import stun
from aioice.stun import TransactionTimeout
from .ai_orchestrator import AIOrchestrator
from .conversation import ConversationPhase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch aioice Transaction retries to avoid crashes when transports close while
# timers are still scheduled (observed AttributeError: 'NoneType' object has no
# attribute 'call_exception_handler' from Transaction.__retry).
# This safely aborts retries once the transport is gone.
# ---------------------------------------------------------------------------
def _safe_stun_retry(self):
    # Access mangled/private members explicitly
    fut = getattr(self, "_Transaction__future", None)
    if fut is None or fut.done():
        return

    tries = getattr(self, "_Transaction__tries", 0)
    tries_max = getattr(self, "_Transaction__tries_max", 0)

    if tries >= tries_max:
        if not fut.done():
            fut.set_exception(TransactionTimeout())
        return

    proto = getattr(self, "_Transaction__protocol", None)
    if proto is None or getattr(proto, "transport", None) is None:
        # Transport already torn down; resolve the future to stop retries
        if not fut.done():
            fut.set_exception(TransactionTimeout())
        return

    try:
        proto.send_stun(
            getattr(self, "_Transaction__request"),
            getattr(self, "_Transaction__addr"),
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.debug("Skipping STUN retry; transport closed: %s", exc)
        if not fut.done():
            fut.set_exception(TransactionTimeout())
        return

    loop = asyncio.get_event_loop()
    delay = getattr(self, "_Transaction__timeout_delay")
    handle = loop.call_later(delay, self._Transaction__retry)
    setattr(self, "_Transaction__timeout_handle", handle)
    setattr(self, "_Transaction__timeout_delay", delay * 2)
    setattr(self, "_Transaction__tries", tries + 1)


# Monkey-patch the retry method once on import
stun.Transaction._Transaction__retry = _safe_stun_retry

class AIOutgoingAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.q = asyncio.Queue()
        self._timestamp = 0
        self.sample_rate = 48000
        self.ptime = 0.02 # 20ms
        self.samples_per_frame = int(self.sample_rate * self.ptime) # 960

    async def recv(self):
        # Handle pacing
        # In a real app we'd adhere to wall clock, for now handling simple sleep
        await asyncio.sleep(self.ptime)

        # Create silent frame by default
        # s16 = 16-bit signed integer
        data = np.zeros(self.samples_per_frame, dtype=np.int16)

        # Check for audio in queue
        if not self.q.empty():
            chunk = await self.q.get()
            # If chunk is raw bytes, we need to ensure it matches our frame size or buffer it.
            # For this MVP, we assume MockTTS returns a big blob, and we chunk it here or earlier.
            # Ideally the consumer of the queue puts ready-to-play frames or we implement a jitter buffer.
            # Simplified: if we have a big chunk, we might just assume 48kHz s16 mono and slice it?
            # Or better: The MockTTS returns a whole WAV/bytes. We need to stream it.
            
            # Re-implementation: The orchestrator should probably push small chunks, or we buffer here.
            # For simplicity: If queue has data, we treat it as a frame (960 samples * 2 bytes = 1920 bytes).
            # If MockTTS returned 10000 bytes, we need to split it.
            
            # We'll assume the external pusher splits it, OR we handle it here.
            # Let's handle generic bytes.
            if len(chunk) > 0:
                # Naive assumption: chunk is exactly 1 frame or we truncate/pad
                # Real implementation needs a byte buffer.
                required_bytes = self.samples_per_frame * 2
                if len(chunk) >= required_bytes:
                    data = np.frombuffer(chunk[:required_bytes], dtype=np.int16)
                else:
                    # Pad with silence
                    padded = chunk + b'\x00' * (required_bytes - len(chunk))
                    data = np.frombuffer(padded, dtype=np.int16)

        frame = av.AudioFrame.from_ndarray(data.reshape(1, -1), format='s16', layout='mono')
        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        
        self._timestamp += self.samples_per_frame
        return frame

class WebRTCManager:
    def __init__(self, room_id, signal_callback, notify_callback=None):
        self.room_id = room_id
        self.signal_callback = signal_callback
        # notify_callback: async callable(dict) -> None; used to emit UI events (e.g., transcripts)
        self.notify_callback = notify_callback
        self.pcs = set()
        self.orchestrator = AIOrchestrator()
        self.running = True
        self.ai_is_speaking = False  # Flag to prevent echo/feedback during AI speech
        # Add lock to serialize incoming utterance processing
        self._utterance_processing_lock = asyncio.Lock()
        # Add lock to protect controller initialization (prevents concurrent start() calls)
        self._controller_init_lock = asyncio.Lock()
        # Flag to track if controller has been initialized
        self._controller_initialized = False
        # Track all spawned tasks for proper cleanup
        self._tasks = set()
        # Track last fully-consumed utterance to avoid duplicate STT processing
        self._last_consumed_utterance_id = None
        # Flag to track when manager is closing (prevents ice-candidate processing race)
        self._closing = False

    async def handle_offer(self, sdp, type_):
        # CRITICAL: Only handle the FIRST offer per manager
        # Multiple offers (from multiple WebSocket connections) should reuse the same PC
        if hasattr(self, 'pc') and self.pc is not None:
            logger.info(f"PC already exists for room {self.room_id}; reusing existing peer connection (idempotent offer handling)")
            # Still need to process this offer on the existing PC
            offer = RTCSessionDescription(sdp=sdp, type=type_)
            await self.pc.setRemoteDescription(offer)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            return {
                'type': 'answer',
                'sdp': self.pc.localDescription.sdp,
                'sdpType': 'answer'
            }
        
        # Create peer connection ONCE per manager
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            pass

        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind}")
            if track.kind == "audio":
                task = asyncio.ensure_future(self.process_incoming_audio(track))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        # Create AI Audio Track and add to PC
        self.ai_track = AIOutgoingAudioTrack()
        pc.addTrack(self.ai_track)

        # Create ConversationSession + Controller ONCE per room (idempotent)
        # Use lock to ensure initialization happens exactly once even if handle_offer is called concurrently
        async with self._controller_init_lock:
            # Check flag inside lock - another concurrent handle_offer may have already initialized
            if self._controller_initialized:
                logger.info(f"Controller already initialized for room {self.room_id}; skipping duplicate initialization")
            else:
                # If already initialized (from previous offer), reuse existing session/controller
                if not hasattr(self, 'session') or self.session is None:
                    try:
                        from .conversation import ConversationSession, ConversationController
                        from ai_agent.models import Call
                        from ai_personas.models import Persona
                        
                        # CRITICAL: Load persona from Room's Call record using Channels-aware database_sync_to_async
                        persona_slug = None
                        try:
                            @database_sync_to_async
                            def get_call_persona():
                                call = Call.objects.filter(room__id=self.room_id).order_by('-started_at').first()
                                if call and call.persona:
                                    return call.persona.slug
                                return None
                            
                            persona_slug = await get_call_persona()
                            if persona_slug:
                                logger.info(f"Loaded persona '{persona_slug}' from Call for room {self.room_id}")
                            else:
                                logger.warning(f"No Call with persona found for room {self.room_id}; falling back to default")
                        except Exception as e:
                            logger.error(f"Error fetching Call for room {self.room_id}: {e}", exc_info=True)
                        
                        # Fallback to default if no persona resolved
                        if not persona_slug:
                            persona_slug = 'default'
                            logger.info(f"Using default persona for room {self.room_id}")
                        
                        # Fetch persona config deterministically using Channels-aware database_sync_to_async
                        try:
                            @database_sync_to_async
                            def get_persona_config():
                                persona_obj = Persona.objects.get(slug=persona_slug)
                                return persona_obj.config
                            
                            persona_config = await get_persona_config()
                            logger.info(f"Initialized persona '{persona_slug}' with config keys: {list(persona_config.keys())}")
                        except Persona.DoesNotExist:
                            logger.error(f"Persona '{persona_slug}' does not exist in database")
                            raise ValueError(f"Persona '{persona_slug}' not found")
                        
                        # Create session and controller with resolved persona
                        self.session = ConversationSession(self.room_id, persona_slug, persona_config)
                        # Pass notify callback to controller so AI messages can be broadcast
                        self.controller = ConversationController(
                            self.session,
                            self.orchestrator,
                            self.queue_audio_output,
                            notify_callback=self._notify
                        )
                        
                        # Start controller (speak first) asynchronously (don't block negotiation)
                        asyncio.ensure_future(self.controller.start())
                        logger.info(f"Conversation controller started for room {self.room_id} with persona {persona_slug}")
                        
                        # Mark as initialized
                        self._controller_initialized = True
                        
                    except Exception as e:
                        logger.error(f"Failed to initialize conversation controller for room {self.room_id}: {e}", exc_info=True)

        offer = RTCSessionDescription(sdp=sdp, type=type_)
        await pc.setRemoteDescription(offer)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        self.pc = pc

        return {
            'type': 'answer',
            'sdp': pc.localDescription.sdp,
            'sdpType': 'answer'
        }

    async def process_incoming_audio(self, track):
        """
        Reads audio frames, runs VAD, buffers utterance, calls ConversationController or Orchestrator.
        """
        logger.info(f"Starting audio processing for room {self.room_id}")
        
        # Wait for controller to be initialized before processing
        max_wait = 50
        for _ in range(max_wait):
            if self.controller is not None:
                logger.info(f"ConversationController is ready; starting audio processing for room {self.room_id}")
                break
            await asyncio.sleep(0.1)
        
        if self.controller is None:
            logger.warning(f"Audio processing started but no controller initialized for room {self.room_id}; will use fallback orchestrator mode")
        
        buffer = bytearray()
        silence_frames = 0
        speaking = False
        frame_count = 0
        voiced_frames = 0
        total_frames_in_utt = 0
        recent_energy_window = []
        
        # CRITICAL: Post-AI-speech grace period to avoid echo/noise false triggers
        # When AI finishes speaking, ignore all audio for N frames to let echo/noise settle
        # This prevents catching tail-end of AI audio or user ambient noise as a new utterance
        post_ai_grace_frames = 0  # Countdown timer
        POST_AI_GRACE_MS = int(os.environ.get("VAD_POST_AI_GRACE_MS", "800"))  # 800ms grace period
        POST_AI_GRACE_FRAMES = int(POST_AI_GRACE_MS / (0.02 * 1000))  # Convert to frame count
        
        # Consecutive voiced frame counter for start detection
        # Require N consecutive frames above threshold before accepting speech start
        # This prevents single-frame noise spikes from triggering false utterances
        consecutive_voiced = 0
        MIN_CONSECUTIVE_VOICED = int(os.environ.get("VAD_MIN_CONSECUTIVE_VOICED", "3"))  # 60ms continuous speech
        
        # Track previous ai_is_speaking state to detect transitions
        prev_ai_speaking = False
        
        # VAD Parameters (tuned for Sarvam AI 16kHz optimization with adaptive thresholds)
        # Two-tier system: high threshold for start detection, lower for continuation
        # This prevents mid-sentence cutoff when user pauses or speaks quieter words
        FRAME_DURATION = 0.02 # 20ms
        START_THRESHOLD = int(os.environ.get("VAD_START_THRESHOLD", "500"))  # Start of speech
        CONTINUE_THRESHOLD = int(os.environ.get("VAD_CONTINUE_THRESHOLD", "250"))  # Mid-sentence continuation
        # INCREASED: 800ms to allow natural user pauses without cutoff (from 600ms)
        # User may pause mid-thought; we want complete sentences not fragments
        SILENCE_DURATION_MS = int(os.environ.get("VAD_SILENCE_DURATION_MS", "800"))
        SILENCE_FRAME_COUNT = int(SILENCE_DURATION_MS / (FRAME_DURATION * 1000))
        ENERGY_WINDOW_SIZE = 10  # Track recent 200ms of energy
        
        # Additional gating to avoid calling STT on noise/very short utterances
        # TUNED: Lowered thresholds to capture normal speech ("JavaScript", "yes", etc.)
        # Users speak naturally with pauses/unvoiced consonants, not all frames are voiced
        MIN_UTTER_MS = int(os.environ.get("VAD_MIN_UTTER_MS", "300"))  # 300ms allows short responses
        MIN_VOICED_FRAMES = int(os.environ.get("VAD_MIN_VOICED_FRAMES", "4"))  # 4 frames = 80ms min spoken sound
        MIN_VOICED_RATIO = float(os.environ.get("VAD_MIN_VOICED_RATIO", "0.08"))  # 8% voiced (natural speech has pauses)
        
        logger.info(f"VAD settings: start={START_THRESHOLD}, continue={CONTINUE_THRESHOLD}, silence={SILENCE_DURATION_MS}ms (REDUCED for <30s audio limit)")

        try:
            while self.running:
                try:
                    frame = await track.recv()
                    frame_count += 1
                except Exception as e:
                    logger.info(f"Audio track ended: {e}")
                    break
                
                # Convert to numpy for energy calculation
                # frame.to_ndarray() returns shape (channels, samples), e.g. (1, 960)
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr[0]
                arr = arr.astype(np.int16, copy=False)
                energy = np.max(np.abs(arr))
                
                # Log energy every 50 frames (~1 second) for debugging
                if frame_count % 50 == 0:
                    logger.info(f"Audio frame {frame_count}: energy={energy}, speaking={speaking}, buffer_size={len(buffer)}")
                
                # Raw bytes for buffer (s16le mono 48kHz)
                raw_bytes = arr.tobytes()
                
                # CRITICAL: Skip VAD processing if AI is currently speaking (prevents echo/feedback)
                # This must be checked BEFORE any speech detection logic
                if self.ai_is_speaking:
                    logger.debug(f"VAD skipped (AI speaking) - energy={energy}")
                    # Reset all VAD state when AI is speaking to ensure clean slate
                    speaking = False
                    buffer = bytearray()
                    silence_frames = 0
                    consecutive_voiced = 0
                    voiced_frames = 0
                    total_frames_in_utt = 0
                    prev_ai_speaking = True
                    continue
                
                # CRITICAL: Detect AI->User transition and start grace period
                # When ai_is_speaking changes from True->False, start countdown to ignore echo/noise
                if prev_ai_speaking and not self.ai_is_speaking:
                    post_ai_grace_frames = POST_AI_GRACE_FRAMES
                    logger.info(f"🕐 AI stopped speaking - starting {POST_AI_GRACE_MS}ms grace period ({POST_AI_GRACE_FRAMES} frames)")
                    prev_ai_speaking = False
                
                # CRITICAL: Post-AI-speech grace period countdown
                # After AI stops, ignore all audio for N frames to avoid echo/noise triggering new utterance
                if post_ai_grace_frames > 0:
                    post_ai_grace_frames -= 1
                    if post_ai_grace_frames % 10 == 0:  # Log every 200ms
                        logger.debug(f"VAD grace period active ({post_ai_grace_frames} frames remaining) - energy={energy}")
                    continue
                
                # Update rolling energy window for adaptive tracking
                recent_energy_window.append(energy)
                if len(recent_energy_window) > ENERGY_WINDOW_SIZE:
                    recent_energy_window.pop(0)
                
                # Adaptive two-tier VAD: different thresholds for start vs continuation
                # Use high threshold for initial detection, lower threshold once speaking to allow mid-sentence pauses
                active_threshold = CONTINUE_THRESHOLD if speaking else START_THRESHOLD
                
                if energy > active_threshold:
                    if not speaking:
                        # CRITICAL: Require consecutive voiced frames before accepting speech start
                        # This prevents single-frame noise spikes from triggering false utterances
                        consecutive_voiced += 1
                        if consecutive_voiced < MIN_CONSECUTIVE_VOICED:
                            logger.debug(f"Consecutive voiced {consecutive_voiced}/{MIN_CONSECUTIVE_VOICED} - energy={energy}")
                            continue
                        # Threshold met: accept as speech start
                        logger.info(f"Speech detected! Energy={energy} (threshold={START_THRESHOLD}, consecutive={consecutive_voiced})")
                        consecutive_voiced = 0  # Reset for next utterance
                        # reset counters at start of utterance
                        voiced_frames = 0
                        total_frames_in_utt = 0
                        recent_energy_window = [energy]  # Reset window
                    else:
                        # Already speaking: reset consecutive counter (not used during continuation)
                        consecutive_voiced = 0
                    speaking = True
                    silence_frames = 0
                    buffer.extend(raw_bytes)
                    total_frames_in_utt += 1
                    voiced_frames += 1
                else:
                    # Below threshold: reset consecutive voiced counter if not speaking
                    if not speaking:
                        consecutive_voiced = 0
                    if speaking:
                        # Check if this is truly silence or just a brief dip
                        # Use recent energy average to avoid cutting on momentary quiet syllables
                        avg_recent = sum(recent_energy_window) / max(len(recent_energy_window), 1)
                        if avg_recent < CONTINUE_THRESHOLD:
                            silence_frames += 1
                        else:
                            # Recent frames were loud enough; reset silence counter (brief dip)
                            silence_frames = 0
                        
                        buffer.extend(raw_bytes)
                        total_frames_in_utt += 1
                        
                        if silence_frames > SILENCE_FRAME_COUNT:
                            # End of utterance
                            logger.info(f"End of utterance detected after {len(buffer)} bytes. Processing...")
                            
                            # Copy buffer before resetting
                            audio_data = bytes(buffer)
                            buffer = bytearray()
                            speaking = False
                            silence_frames = 0

                            # Utterance quality gating: drop low-information/noise segments
                            utter_ms = int(total_frames_in_utt * (FRAME_DURATION * 1000))
                            voiced_ratio = (voiced_frames / max(total_frames_in_utt, 1)) if total_frames_in_utt else 0.0
                            if utter_ms < MIN_UTTER_MS or voiced_frames < MIN_VOICED_FRAMES or voiced_ratio < MIN_VOICED_RATIO:
                                logger.info(f"🛑 Dropping low-information utterance (voiced={voiced_frames}, total={total_frames_in_utt}, ms={utter_ms}, ratio={voiced_ratio:.2f})")
                                voiced_frames = 0
                                total_frames_in_utt = 0
                                continue
                            
                            # reset for next utterance window
                            voiced_frames = 0
                            total_frames_in_utt = 0

                            # Guard: only process when controller expects input
                            expected_id = getattr(getattr(self, 'session', None), 'expected_user_utterance_id', None)
                            current_phase = getattr(getattr(self, 'session', None), 'phase', None)

                            if current_phase and current_phase != ConversationPhase.WAIT_FOR_USER:
                                logger.info(f"🛑 Dropping buffered audio because phase is {current_phase}, not WAIT_FOR_USER")
                                continue

                            if expected_id and expected_id == self._last_consumed_utterance_id:
                                logger.info(f"🛑 Duplicate end-of-utterance for already processed id {expected_id}; skipping")
                                continue

                            if current_phase == ConversationPhase.WAIT_FOR_USER and not expected_id:
                                logger.warning("🛑 No expected_user_utterance_id set while waiting; dropping buffer to avoid duplicate STT")
                                continue
                            
                            logger.info(f"🔊 Calling STT for {len(audio_data)} bytes of audio...")
                            
                            try:
                                transcript = await self.orchestrator.stt.transcribe(audio_data)
                                logger.info(f"✅ STT returned: '{transcript}' (length: {len(transcript)})")
                            except Exception as e:
                                logger.error(f"❌ STT failed: {e}", exc_info=True)
                                transcript = ""

                            if not transcript.strip():
                                logger.warning(f"⚠️ Empty transcript, skipping utterance")
                                continue
                            
                            logger.info(f"📝 Transcript received: '{transcript}' - forwarding to controller")

                            # Freeze the utterance id we intend to consume
                            # CRITICAL: Re-check expected_id hasn't changed (guards against race condition)
                            utterance_id = expected_id or str(uuid.uuid4())
                            
                            # Validate the expected_id is still valid (hasn't been updated to next utterance)
                            current_expected = getattr(getattr(self, 'session', None), 'expected_user_utterance_id', None)
                            if expected_id and current_expected and expected_id != current_expected:
                                logger.warning(f"⚠️ Utterance ID mismatch: expected_id changed from {expected_id} to {current_expected} during STT")
                                logger.warning(f"   Skipping stale transcript to prevent response mix-up")
                                continue
                            
                            # Emit a user transcript event to UI before controller handling
                            await self._notify({
                                'type': 'transcript',
                                'role': 'user',
                                'text': transcript,
                                'utteranceId': utterance_id,
                                'roomId': self.room_id,
                                'timestamp': time.time(),
                            })

                            if hasattr(self, 'controller') and self.controller:
                                logger.info(f"🎮 Controller exists, calling handle_user_utterance...")
                                try:
                                    # Use lock to serialize utterance processing - one at a time
                                    async with self._utterance_processing_lock:
                                        await self.controller.handle_user_utterance(transcript, utterance_id)
                                        self._last_consumed_utterance_id = utterance_id
                                    logger.info(f"✅ Controller processing completed")
                                except Exception as e:
                                    logger.error(f"❌ Controller handling failed: {e}", exc_info=True)
                            else:
                                logger.warning("⚠️ No controller found, skipping processing")
                                persona_slug = "default"
                                try:
                                    from ai_agent.models import Call
                                    
                                    @database_sync_to_async
                                    def get_fallback_persona():
                                        call = Call.objects.filter(room__id=self.room_id).order_by('-started_at').first()
                                        if call and call.persona:
                                            return call.persona.slug
                                        return "default"
                                    
                                    persona_slug = await get_fallback_persona()
                                except Exception:
                                    pass

                                result = await self.orchestrator.handle_utterance(self.room_id, audio_data, persona_slug)
                                if result and result.get('tts_audio'):
                                    await self.queue_audio_output(result['tts_audio'])
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")

    async def queue_audio_output(self, audio_bytes, turn_id: str | None = None):
        """
        Split big audio blob into 20ms chunks and push to queue.
        Assumes audio_bytes is 48kHz s16 mono (matching track expectations).
        Sets ai_is_speaking flag to prevent echo/feedback during playback.
        
        CRITICAL: This must block until audio is fully played to ensure
        VAD doesn't pick up our own TTS output.
        """
        logger.info(f"🔇 Queueing {len(audio_bytes)} bytes of audio for playback...")
        
        # FIRST: Set flag BEFORE pushing any audio to queue
        self.ai_is_speaking = True
        # Notify UI that playback is starting for this turn (if available)
        try:
            if turn_id:
                await self._notify({
                    'type': 'ai_playback',
                    'status': 'start',
                    'turnId': turn_id,
                    'roomId': self.room_id,
                    'timestamp': time.time(),
                })
        except Exception:
            pass
        
        # Calculate duration FIRST so we know how long to wait
        duration_seconds = len(audio_bytes) / (48000 * 2 * 1)
        
        try:
            # Push audio chunks to track
            CHUNK_SIZE = 1920  # 960 samples * 2 bytes = 20ms at 48kHz
            chunks_queued = 0
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i:i+CHUNK_SIZE]
                if chunk:  # Only queue non-empty chunks
                    await self.ai_track.q.put(chunk)
                    chunks_queued += 1
            
            logger.info(f"🔊 Queued {chunks_queued} chunks ({duration_seconds:.2f}s of audio) - VAD disabled to prevent echo")
            
            # CRITICAL: Wait for audio to finish playing before re-enabling VAD
            # Add 15% buffer to ensure playback completes + any codec/network jitter
            wait_time = duration_seconds * 1.15
            await asyncio.sleep(wait_time)
            
        finally:
            # ALWAYS re-enable VAD, even if there was an error
            self.ai_is_speaking = False
            logger.info(f"🎤 AI finished speaking - VAD will re-enable after grace period")
            # Notify UI that playback ended for this turn
            try:
                if turn_id:
                    await self._notify({
                        'type': 'ai_playback',
                        'status': 'end',
                        'turnId': turn_id,
                        'roomId': self.room_id,
                        'timestamp': time.time(),
                    })
            except Exception:
                pass

    async def handle_candidate(self, candidate_data):
        # GUARD: Skip all candidate handling if manager is closing
        if self._closing:
            logger.debug(f"Ignoring ICE candidate; manager is closing for room {self.room_id}")
            return
        
        if not hasattr(self, 'pc') or not self.pc:
            return

        try:
             sdp = candidate_data.get('candidate')
             sdp_mid = candidate_data.get('sdpMid')
             sdp_mline_index = candidate_data.get('sdpMLineIndex')
             
             if sdp:
                 parts = sdp.split()
                 if len(parts) >= 8:
                     from aiortc import RTCIceCandidate
                     cand = RTCIceCandidate(
                        foundation=parts[0].split(':')[1],
                        component=int(parts[1]),
                        protocol=parts[2],
                        priority=int(parts[3]),
                        ip=parts[4],
                        port=int(parts[5]),
                        type=parts[7],
                        sdpMid=sdp_mid,
                        sdpMLineIndex=sdp_mline_index
                     )
                     await self.pc.addIceCandidate(cand)
        
        except Exception as e:
            logger.debug(f"Failed to handle ICE candidate (may be closing): {e}")

    async def close(self):
        logger.info(f"🛑 Closing WebRTCManager for room {self.room_id}")
        self.running = False
        # Signal that we're closing to prevent ICE candidate race conditions
        self._closing = True
        
        # Cancel all spawned tasks
        logger.info(f"Cancelling {len(self._tasks)} background tasks for room {self.room_id}")
        for task in list(self._tasks):
            try:
                task.cancel()
            except Exception as e:
                logger.error(f"Error cancelling task: {e}")
        
        # Wait briefly for tasks to finish cancelling
        if self._tasks:
            try:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error waiting for tasks to finish: {e}")
        
        self._tasks.clear()
        
        # Close all peer connections
        # Iterate over a snapshot to avoid 'set changed size during iteration'
        for pc in list(self.pcs):
            try:
                logger.info(f"Closing peer connection {pc} for room {self.room_id}")
                await pc.close()
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")
            finally:
                self.pcs.discard(pc)
        
        self.pcs.clear()
        if hasattr(self, 'pc') and self.pc:
            try:
                await self.pc.close()
            except Exception as e:
                logger.error(f"Error closing main PC: {e}")
            self.pc = None
        
        logger.info(f"✅ WebRTCManager closed for room {self.room_id}")

    async def _notify(self, message: dict):
        """Send a UI event if a notify callback is available."""
        try:
            if self.notify_callback:
                await self.notify_callback(message)
        except Exception as e:
            logger.debug(f"Notify callback failed: {e}")


