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
from .audio_conditioning import AudioConditioner
from .persona_behaviors import get_behavior_for_persona
from .barge_in_state_machine import BargeinStateMachine, BargeinState
from .stop_word_interruption import StopWordInterruptor, InterruptionTier
from typing import Deque
from collections import deque

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
        # TTS cancellation support
        self._tts_cancel_event = asyncio.Event()
        self._tts_playback_task: asyncio.Task | None = None
        # Audio conditioning pipeline
        self.audio_conditioner = AudioConditioner(sample_rate=48000)
        # Rolling overlapping buffers for transcript stability
        self.rolling_transcripts: Deque[str] = deque(maxlen=3)  # Keep last 3 transcripts
        # Guard window after AI playback starts (barge-in ignored during this window)
        self._ai_playback_guard_deadline: float = 0.0
        # Barge-in state machine (will be initialized after persona is loaded)
        self.barge_in_machine = None
        # Stop-word interruption (semantic control during AI speech)
        self.stop_interruptor = StopWordInterruptor(check_interval_ms=500)
        self._stopword_buffer = bytearray()
        self._ai_playback_started_at: float | None = None
        # Track consecutive empty stop-check STT results during AI playback
        self._stopcheck_empty_count: int = 0
        self._last_stopcheck_ts: float = 0.0
        # Cooldown to avoid thrashing stop-check STT on ultra-low SNR during AI speech
        self._stopcheck_low_snr_until: float = 0.0
        # Debounce repeated utterance finalizations
        self._last_finalize_ts: float = 0.0
        # Conversation/Interview controller (initialized in handle_offer)
        self.controller = None

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
                                return persona_obj.upgraded_config  # Auto-upgrade old configs
                            
                            persona_config = await get_persona_config()
                            logger.info(f"Initialized persona '{persona_slug}' with config keys: {list(persona_config.keys())}")
                        except Persona.DoesNotExist:
                            logger.error(f"Persona '{persona_slug}' does not exist in database")
                            raise ValueError(f"Persona '{persona_slug}' not found")
                        
                        # Create session and controller with resolved persona
                        self.session = ConversationSession(self.room_id, persona_slug, persona_config)

                        # STRICT ISOLATION: route interviewer personas to InterviewerController
                        meta = persona_config.get('metadata', {}) if persona_config else {}
                        source_template = meta.get('source_template_id')
                        flow_mode = persona_config.get('flow')

                        is_interviewer_persona = (
                            persona_slug == 'technical-interviewer' or
                            source_template == 'technical-interviewer' or
                            flow_mode == 'interview'
                        )

                        if is_interviewer_persona:
                            from ai_agent.models import Call
                            from ai_agent.interviewer.controller import InterviewerController

                            @database_sync_to_async
                            def get_call_id_for_room():
                                try:
                                    return Call.objects.filter(room__id=self.room_id).order_by('-started_at').values_list('id', flat=True).first()
                                except Exception:
                                    return None

                            call_id = await get_call_id_for_room()
                            if not call_id:
                                logger.error("No Call found for interviewer controller; cannot start interviewer flow")
                                raise ValueError("Call not initialized for room")

                            try:
                                self.controller = InterviewerController(
                                    self.room_id,
                                    call_id,
                                    persona_slug,
                                    persona_config,
                                    self.orchestrator,
                                    self.queue_audio_output,
                                    notify_callback=self._notify,
                                    session=self.session,
                                )
                                logger.info(
                                    "✅ Routed persona to InterviewerController | slug=%s | template=%s | flow=%s",
                                    persona_slug,
                                    source_template,
                                    flow_mode,
                                )
                            except Exception as e:
                                logger.error(f"❌ Failed to initialize InterviewerController: {e}", exc_info=True)
                                # Fallback to normal conversation controller instead of crashing
                                self.controller = ConversationController(
                                    self.room_id,
                                    persona_slug,
                                    persona_config,
                                    self.orchestrator,
                                    self.queue_audio_output,
                                    notify_callback=self._notify,
                                    session=self.session,
                                )
                                logger.info(
                                    "⚠️ Fell back to ConversationController due to initialization error | slug=%s",
                                    persona_slug,
                                )
                        else:
                            # Pass notify callback to controller so AI messages can be broadcast
                            self.controller = ConversationController(
                                self.session,
                                self.orchestrator,
                                self.queue_audio_output,
                                notify_callback=self._notify
                            )
                        
                        # Initialize persona-aware barge-in behavior
                        barge_behavior = get_behavior_for_persona(persona_slug, persona_config)
                        self.barge_in_machine = BargeinStateMachine(barge_behavior)
                        logger.info(f"✅ Initialized intelligent barge-in for persona '{persona_slug}'")
                        
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
        
        The new pipeline enforces strict turn-taking with explicit gating during AI speech,
        conservative barge-in, and a single STT call per finalized user utterance.
        """
        logger.info(f"Starting audio processing for room {self.room_id}")

        capture_sample_rate: int | None = None
        try:
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)
        except Exception as e:
            logger.error(f"Failed to init AudioResampler: {e}")
            resampler = None

        max_wait = 50
        for _ in range(max_wait):
            if self.controller is not None:
                logger.info(f"ConversationController is ready; starting audio processing for room {self.room_id}")
                break
            await asyncio.sleep(0.1)

        if self.controller is None:
            logger.warning(f"Audio processing started but no controller initialized for room {self.room_id}; will use fallback orchestrator mode")

        FRAME_DURATION = 0.02
        BASE_START = int(os.environ.get("VAD_START_THRESHOLD", "500"))
        BASE_CONTINUE = int(os.environ.get("VAD_CONTINUE_THRESHOLD", "250"))
        SILENCE_DURATION_MS = 2500
        SILENCE_FRAME_COUNT = int(SILENCE_DURATION_MS / (FRAME_DURATION * 1000))
        MIN_BUFFER_MS = 1500
        MIN_BUFFER_BYTES = int(MIN_BUFFER_MS * 48000 * 2 / 1000)
        MAX_UTTERANCE_MS = int(os.environ.get("VAD_MAX_UTTERANCE_MS", "30000"))
        MAX_UTTERANCE_BYTES = int(MAX_UTTERANCE_MS * 48000 * 2 / 1000)

        START_SNR_DB = float(os.environ.get("VAD_START_SNR_DB", "6"))  # REDUCED from 10 to be more sensitive
        MIN_CONSECUTIVE_START = int(os.environ.get("VAD_MIN_CONSECUTIVE_START", "3"))  # REDUCED from 6 (60ms vs 120ms)
        POST_AI_GRACE_MS = int(os.environ.get("VAD_POST_AI_GRACE_MS", "1200"))
        POST_AI_GRACE_FRAMES = int(POST_AI_GRACE_MS / (FRAME_DURATION * 1000))

        BARGE_PEAK_MIN = int(os.environ.get("VAD_BARGE_PEAK_MIN", "1800"))
        BARGE_MIN_FRAMES = int(os.environ.get("VAD_BARGE_MIN_FRAMES", "10"))
        BARGE_SNR_DB = float(os.environ.get("VAD_BARGE_SNR_DB", "12"))

        STT_MIN_DURATION_MS = 400
        STT_MIN_RMS = 0.003
        STT_MIN_SNR_DB = 3.0

        buffer = bytearray()
        silence_frames = 0
        speaking = False
        frame_count = 0
        consecutive_start_frames = 0
        # OLD: consecutive_barge_frames = 0 (replaced by state machine)
        post_ai_grace_frames = 0
        speech_start_time = None
        utterance_in_flight = False
        current_utterance_id = None
        prev_ai_speaking = False

        logger.info(
            f"VAD settings: start>={BASE_START}, continue>={BASE_CONTINUE}, silence={SILENCE_DURATION_MS}ms, "
            f"min_buffer={MIN_BUFFER_MS}ms, barge_min_peak={BARGE_PEAK_MIN}, barge_frames={BARGE_MIN_FRAMES}"
        )
        logger.info(
            f"VAD sensitivity: START_SNR_DB={START_SNR_DB}dB, MIN_CONSECUTIVE_START={MIN_CONSECUTIVE_START} frames (60ms), "
            f"Grace period={POST_AI_GRACE_MS}ms with 1.3x multiplier"
        )

        try:
            while self.running:
                try:
                    frame = await track.recv()
                    frame_count += 1
                except Exception as e:
                    logger.info(f"Audio track ended: {e}")
                    break

                try:
                    if resampler:
                        resampled_frames = resampler.resample(frame)
                        if isinstance(resampled_frames, list):
                            if len(resampled_frames) == 0:
                                continue
                            resampled_frame = resampled_frames[0]
                        else:
                            resampled_frame = resampled_frames
                        arr = resampled_frame.to_ndarray()
                        capture_sample_rate = 48000
                    else:
                        arr = frame.to_ndarray()
                        capture_sample_rate = getattr(frame, "sample_rate", 48000) or 48000
                except Exception as e:
                    logger.warning(f"Frame resample failed; using raw frame: {e}")
                    arr = frame.to_ndarray()
                    capture_sample_rate = getattr(frame, "sample_rate", 48000) or 48000

                if arr.ndim > 1:
                    arr = arr[0]
                arr = arr.astype(np.int16, copy=False)

                energy_peak = int(np.max(np.abs(arr)))
                energy_rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                raw_bytes = arr.tobytes()

                samples_float = arr.astype(np.float32) / 32767.0
                # Only learn noise when AI is not speaking to avoid bias from playback
                self.audio_conditioner.noise_estimator.update(samples_float, is_speech=self.ai_is_speaking or speaking)
                noise_floor = self.audio_conditioner.noise_estimator.get_noise_floor()
                snr_db = 20 * np.log10(max(energy_rms, 1e-6) / max(noise_floor, 1e-6))

                if frame_count % 50 == 0:
                    logger.info(
                        f"Audio frame {frame_count}: peak={energy_peak}, rms={energy_rms:.4f}, snr={snr_db:.1f} dB, "
                        f"speaking={speaking}, buffer_size={len(buffer)}"
                    )

                if prev_ai_speaking and not self.ai_is_speaking:
                    post_ai_grace_frames = POST_AI_GRACE_FRAMES
                    logger.info(
                        f"🕐 AI stopped speaking - starting {POST_AI_GRACE_MS}ms grace period ({POST_AI_GRACE_FRAMES} frames)"
                    )

                prev_ai_speaking = self.ai_is_speaking
                expected_id = getattr(getattr(self, "session", None), "expected_user_utterance_id", None)

                # ========== INTELLIGENT BARGE-IN: TWO-STAGE INTERRUPTION PIPELINE ==========
                # Stage 1: Speech Validation (sustained energy check)
                # Stage 2: Query Validation (preliminary STT + heuristics)
                # Only accept interruption if BOTH stages pass
                # ==========================================================================
                
                # DIAGNOSTIC: Log AI speaking state every 50 frames to debug why stop-check doesn't run
                if frame_count % 50 == 0:
                    current_time = time.time()
                    guard_remaining = max(0, self._ai_playback_guard_deadline - current_time)
                    logger.info(
                        f"🔍 [DIAGNOSTIC] Frame {frame_count}: ai_is_speaking={self.ai_is_speaking}, "
                        f"guard_remaining={guard_remaining:.2f}s, guard_deadline={self._ai_playback_guard_deadline:.2f}, "
                        f"current_time={current_time:.2f}"
                    )
                
                if self.ai_is_speaking:
                    # Guard window: Ignore initial 400ms to prevent false triggers from playback ramp
                    current_time = time.time()
                    if current_time < self._ai_playback_guard_deadline:
                        if frame_count % 50 == 0:
                            logger.info(f"⏱️  [GUARD] Skipping stop-check: still in guard window ({self._ai_playback_guard_deadline - current_time:.2f}s remaining)")
                        continue

                    # Cooldown after repeated low-SNR attempts to avoid spamming STT with AI audio
                    if current_time < self._stopcheck_low_snr_until:
                        if frame_count % 50 == 0:
                            logger.info(
                                f"⏱️  [stop-debug] Cooldown active ({self._stopcheck_low_snr_until - current_time:.2f}s remaining); skipping stop-check"
                            )
                        continue

                    # Always accumulate a short rolling buffer during AI playback for stop-word detection.
                    self._stopword_buffer.extend(raw_bytes)
                    max_stop_bytes = int(2_000 * 48000 * 2 / 1000)  # ~2s
                    if len(self._stopword_buffer) > max_stop_bytes:
                        self._stopword_buffer = self._stopword_buffer[-max_stop_bytes:]

                    # ULTRA-LOW threshold for stop-word detection (300ms = ~1 word)
                    min_stop_bytes = int(300 * 48000 * 2 / 1000)  # 300ms

                    # DEBUG: Log every frame during AI playback to diagnose throttle issues
                    if frame_count % 25 == 0:  # Every 500ms
                        logger.info(
                            f"⏰ [stop-debug] AI speaking | buffer={len(self._stopword_buffer)} bytes ({len(self._stopword_buffer) / (48000 * 2) * 1000:.0f}ms), "
                            f"min_needed={min_stop_bytes / (48000 * 2) * 1000:.0f}ms, will_check={self.stop_interruptor.should_check_now()}"
                        )

                    if len(self._stopword_buffer) >= min_stop_bytes and self.stop_interruptor.should_check_now():
                        stop_buffer_ms = len(self._stopword_buffer) / (48000 * 2) * 1000
                        logger.info(
                            f"🛑 [stop-check] Triggered during AI playback | buffer_ms={stop_buffer_ms:.0f} "
                            f"energy={energy_peak} snr={snr_db:.1f} guard_until={self._ai_playback_guard_deadline:.2f}"
                        )
                        try:
                            prepared_bytes, prep_metrics = self._prepare_audio_for_stt(
                                bytes(self._stopword_buffer), capture_sample_rate
                            )
                            logger.info(
                                "[stop-check] prep metrics | dur=%.0fms rms=%.4f snr=%.1fdB peak=%.3f",
                                prep_metrics.get("duration_ms", 0.0),
                                prep_metrics.get("rms", 0.0),
                                prep_metrics.get("snr_db", 0.0),
                                prep_metrics.get("peak", 0.0),
                            )

                            # Skip stop-word STT when signal is clearly too weak to be meaningful
                            if (
                                prep_metrics.get("snr_db", 0.0) < 8.0
                                or prep_metrics.get("rms", 0.0) < 0.002
                            ):
                                logger.info(
                                    "↩️  [stop-check] Skipping stop-word check (snr=%.1fdB rms=%.4f) — keep AI speaking",
                                    prep_metrics.get("snr_db", 0.0),
                                    prep_metrics.get("rms", 0.0),
                                )
                                # Back off for a short period to avoid hammering STT with AI audio
                                self._stopcheck_low_snr_until = time.time() + 1.0
                                self.stop_interruptor.reset_check_timer()
                                self._stopword_buffer.clear()
                                self._stopcheck_empty_count = 0
                                continue

                            transcript = await self.orchestrator.stt.transcribe(prepared_bytes)

                            # Retry once with boosted audio if transcript is empty but signal seems usable
                            if not transcript.strip() and prep_metrics.get("duration_ms", 0.0) >= 250:
                                boosted_bytes = self._boost_audio_for_stt(prepared_bytes, gain=3.0)
                                logger.info("[stop-check] Empty transcript; retrying stop-word STT with boosted audio")
                                transcript = await self.orchestrator.stt.transcribe(boosted_bytes)
                            # Track empties and time for fallback logic
                            now_ts = time.time()
                            if not transcript.strip():
                                # Increment if within a short window
                                if now_ts - self._last_stopcheck_ts <= 2.0:
                                    self._stopcheck_empty_count += 1
                                else:
                                    self._stopcheck_empty_count = 1
                                self._last_stopcheck_ts = now_ts
                            else:
                                # Reset on any non-empty transcript
                                self._stopcheck_empty_count = 0
                                self._last_stopcheck_ts = now_ts
                            time_in_ai_ms = 0.0
                            if self._ai_playback_started_at:
                                time_in_ai_ms = (time.time() - self._ai_playback_started_at) * 1000

                            match = await self.stop_interruptor.check_for_stop_words(
                                transcript=transcript,
                                buffer_duration_ms=int(prep_metrics.get("duration_ms", 0)),
                                stt_confidence=0.9,
                                time_since_ai_start_ms=time_in_ai_ms,
                                utterance_id=expected_id,
                            )

                            logger.info(
                                "📝 [stop-check] Transcript result | "
                                f"len={len(transcript)} text='{transcript[:80]}' match={match.matched} tier={getattr(match, 'tier', None)}"
                            )

                            if match.matched and match.tier == InterruptionTier.TIER_1_HARD:
                                logger.info(
                                    "🛑 Stop-word detected during AI speech; canceling TTS immediately | "
                                    f"keyword='{match.keyword}' buffer_ms={prep_metrics.get('duration_ms', 0):.0f}"
                                )
                                await self.cancel_tts()
                                self.ai_is_speaking = False
                                self._ai_playback_guard_deadline = 0.0
                                self._stopword_buffer.clear()
                                self.stop_interruptor.reset_check_timer()
                                if self.barge_in_machine:
                                    self.barge_in_machine.reset()

                                # Reset VAD state so the next user audio is captured cleanly
                                speaking = False
                                buffer = bytearray()
                                silence_frames = 0
                                consecutive_start_frames = 0
                                post_ai_grace_frames = 0

                                # Notify UI of control intent (no history/LLM impact)
                                await self._notify(
                                    {
                                        "type": "stop_word",
                                        "keyword": match.keyword or transcript,
                                        "utteranceId": expected_id,
                                        "roomId": self.room_id,
                                        "timestamp": time.time(),
                                    }
                                )

                                if hasattr(self, "controller") and self.controller:
                                    await self.controller._notify_state(ConversationPhase.USER_OVERRIDE)
                                    await self.controller._set_waiting_for_user()
                                continue
                            else:
                                logger.info("ℹ️ [stop-check] No Tier-1 stop-word match; continuing AI playback")
                                # Reset buffer/timer on empty transcripts to avoid repeated low-SNR checks
                                if not transcript.strip():
                                    self._stopword_buffer.clear()
                                    self.stop_interruptor.reset_check_timer()
                        except Exception as e:
                            logger.warning(f"Stop-word detection failed: {e}")

                    # Safety: Fall back to default behavior if barge_in_machine not initialized
                    if self.barge_in_machine is None:
                        logger.warning("⚠️ Barge-in machine not initialized; falling back to simple energy detection")
                        # Fallback to old logic (simplified)
                        barge_peak_threshold = max(BARGE_PEAK_MIN, noise_floor * 32767 * 12)
                        if energy_peak > barge_peak_threshold and snr_db >= BARGE_SNR_DB:
                            logger.info(f"🛑 SIMPLE BARGE-IN (fallback mode): peak={energy_peak}")
                            await self.cancel_tts()
                            self.ai_is_speaking = False
                            speaking = False
                            buffer.clear()
                            silence_frames = 0
                            consecutive_start_frames = 0
                            post_ai_grace_frames = POST_AI_GRACE_FRAMES
                        continue

                    # Process frame through intelligent state machine (runs alongside stop-check)
                    result = self.barge_in_machine.process_frame(
                        energy_peak=energy_peak,
                        snr_db=snr_db,
                        raw_bytes=raw_bytes,
                        noise_floor=noise_floor
                    )

                    action = result["action"]
                    new_state = result["new_state"]
                    reason = result["reason"]

                    # Log state transitions at INFO level for visibility
                    if frame_count % 25 == 0 or new_state != BargeinState.AI_SPEAKING:
                        logger.info(
                            f"🎮 Barge-in state: {new_state.value} | "
                            f"action={action} | energy={energy_peak} | snr={snr_db:.1f} dB"
                        )

                    # Handle state machine actions
                    if action == "accept_barge_in":
                        logger.info("=" * 70)
                        logger.info("✅ BARGE-IN ACCEPTED - Running Stage 2 Query Validation")
                        logger.info(f"   Reason: {reason}")
                        logger.info("=" * 70)

                        # Stage 2: Validate query buffer
                        query_buffer = result.get("query_buffer")
                        if query_buffer:
                            is_valid_query = await self.barge_in_machine.validate_query(
                                query_buffer,
                                self.orchestrator.stt
                            )

                            if is_valid_query:
                                logger.info("✅ Stage 2 PASSED: Query is meaningful, accepting interruption")

                                # Cancel AI speech
                                await self.cancel_tts()
                                self.ai_is_speaking = False

                                # Reset VAD state for fresh user utterance
                                speaking = False
                                buffer.clear()
                                silence_frames = 0
                                consecutive_start_frames = 0
                                post_ai_grace_frames = POST_AI_GRACE_FRAMES

                                # Reset barge-in machine for next detection
                                self.barge_in_machine.reset()

                                # Log final decision
                                logger.info(f"🎤 User interruption accepted, listening for full utterance")
                                continue
                            else:
                                logger.info("❌ Stage 2 FAILED: Query validation rejected")
                                logger.info(f"   Rejection reason: {self.barge_in_machine.context.rejection_reason}")
                                # Reset and continue AI speech
                                self.barge_in_machine.reset()
                                continue
                        else:
                            logger.warning("⚠️  No query buffer available for validation; rejecting by default")
                            self.barge_in_machine.reset()
                            continue

                    elif action == "reject_barge_in":
                        # State machine explicitly rejected (logged inside machine)
                        self.barge_in_machine.reset()
                        continue

                    elif action == "continue_ai":
                        # Still validating; continue AI playback
                        continue

                    else:
                        # Unknown action (shouldn't happen); log and continue
                        logger.warning(f"⚠️  Unknown barge-in action: {action}")
                        continue

                # CRITICAL FIX: Grace period should slightly reduce sensitivity to prevent echo/artifacts
                # BUT must still allow normal human speech (not 3x multiplier which blocks voice!)
                # Previous fix was TOO AGGRESSIVE - users couldn't speak during grace period
                if post_ai_grace_frames > 0:
                    post_ai_grace_frames -= 1
                    if post_ai_grace_frames % 10 == 0:
                        logger.debug(
                            f"Post-AI grace active ({post_ai_grace_frames} frames left) peak={energy_peak}"
                        )
                    # Apply MODEST thresholds during grace period (1.3x multiplier instead of 3.0x)
                    # This filters echo/artifacts without blocking legitimate speech
                    grace_multiplier = 1.3  # Only require 1.3x normal energy during grace (not 3x!)
                    dynamic_start = max(BASE_START * grace_multiplier, noise_floor * 32767 * 10)
                    dynamic_continue = max(BASE_CONTINUE * grace_multiplier, noise_floor * 32767 * 6)
                    # Fall through to VAD detection with slightly stricter thresholds
                else:
                    # Normal VAD thresholds when grace period is over - more sensitive for better voice capture
                    dynamic_start = max(BASE_START, noise_floor * 32767 * 5)  # REDUCED from 8x to 5x
                    dynamic_continue = max(BASE_CONTINUE, noise_floor * 32767 * 3)  # REDUCED from 5x to 3x

                # Thresholds already calculated above based on grace period state

                if not speaking:
                    if energy_peak > dynamic_start and snr_db >= START_SNR_DB:
                        consecutive_start_frames += 1
                    else:
                        consecutive_start_frames = 0

                    if consecutive_start_frames >= MIN_CONSECUTIVE_START:
                        speaking = True
                        speech_start_time = time.time()
                        buffer = bytearray()
                        silence_frames = 0
                        logger.info(
                            f"🎤 Speech START detected (peak={energy_peak}, snr={snr_db:.1f} dB, "
                            f"dyn_start={dynamic_start:.0f})"
                        )
                        # Emit USER_SPEAKING_START event to UI
                        await self._notify({
                            'type': 'speaking_state',
                            'speaker': 'user',
                            'state': 'start',
                            'roomId': self.room_id,
                            'timestamp': time.time(),
                        })
                    continue

                buffer.extend(raw_bytes)

                is_silence_frame = (energy_peak < dynamic_continue) or (snr_db < 2.0)  # REDUCED from 4.0 to 2.0
                if is_silence_frame:
                    silence_frames += 1
                else:
                    silence_frames = 0

                utterance_ready = False
                if silence_frames >= SILENCE_FRAME_COUNT and len(buffer) >= MIN_BUFFER_BYTES:
                    utterance_ready = True
                    logger.info(
                        f"✅ Utterance complete: speech+silence window hit (buffer={len(buffer)} bytes, "
                        f"silence_frames={silence_frames})"
                    )
                elif len(buffer) >= MAX_UTTERANCE_BYTES:
                    utterance_ready = True
                    logger.warning(
                        f"⏳ Max utterance reached ({MAX_UTTERANCE_MS}ms). Forcing STT dispatch (buffer={len(buffer)} bytes)"
                    )

                if not utterance_ready:
                    continue

                audio_data = bytes(buffer)
                buffer = bytearray()
                speaking = False
                silence_frames = 0

                expected_id = getattr(getattr(self, "session", None), "expected_user_utterance_id", None)
                current_phase = getattr(getattr(self, "session", None), "phase", None)

                # FIX: Check barge-in state FIRST before dropping utterances
                # This allows interruptions during AI_SPEAKING phase
                barge_in_decision = None
                if self.barge_in_machine and hasattr(self.barge_in_machine, 'context'):
                    barge_in_state = self.barge_in_machine.context.current_state
                    # If barge-in detected sustained speech during AI_SPEAKING, treat as valid interruption
                    if current_phase == ConversationPhase.AI_SPEAKING and barge_in_state == "BARGE_IN_CANDIDATE":
                        logger.info(f"✅ BARGE-IN DETECTED - User spoke during AI response, will process interruption")
                        barge_in_decision = "interrupt_ai"

                # Only drop if NOT a valid interruption
                if current_phase and current_phase != ConversationPhase.WAIT_FOR_USER and not barge_in_decision:
                    logger.info(f"🛑 Dropping buffered audio because phase is {current_phase}, not WAIT_FOR_USER (and no barge-in)")
                    continue

                if expected_id and expected_id == self._last_consumed_utterance_id:
                    logger.info(f"🛑 Duplicate end-of-utterance for already processed id {expected_id}; skipping")
                    continue

                if current_phase == ConversationPhase.WAIT_FOR_USER and not expected_id:
                    logger.warning(
                        "🛑 No expected_user_utterance_id set while waiting; dropping buffer to avoid duplicate STT"
                    )
                    continue

                if len(audio_data) < MIN_BUFFER_BYTES:
                    # Check if this might be a stop-word BEFORE rejecting
                    # Short utterances are often control commands!
                    try:
                        # Quick STT check on small buffer to detect stop-words
                        quick_prep, quick_metrics = self._prepare_audio_for_stt(audio_data, effective_sample_rate)
                        quick_transcript = await self.orchestrator.stt.transcribe(quick_prep)
                        
                        is_stop_word = any(
                            keyword in quick_transcript.lower()
                            for keyword in ["stop", "wait", "hold", "pause", "quit", "exit"]
                        )
                        
                        if is_stop_word:
                            logger.info(
                                f"✅ [min-buffer-bypass] Small buffer ({len(audio_data)} bytes / "
                                f"{len(audio_data) / (48000 * 2) * 1000:.0f}ms) BUT contains stop-word '{quick_transcript}' - processing"
                            )
                            # Allow processing to continue
                        else:
                            logger.info(
                                f"⏳ Buffer too small at finalize ({len(audio_data)} bytes / "
                                f"{len(audio_data) / (48000 * 2) * 1000:.0f}ms < {MIN_BUFFER_MS}ms); extending capture..."
                            )
                            continue
                    except Exception:
                        # If quick check fails, use original threshold logic
                        logger.info(
                            f"⏳ Buffer too small at finalize ({len(audio_data)} bytes / "
                            f"{len(audio_data) / (48000 * 2) * 1000:.0f}ms < {MIN_BUFFER_MS}ms); extending capture..."
                        )
                        continue

                if utterance_in_flight:
                    logger.warning(
                        f"⚠️ STT already in flight for utterance {current_utterance_id}; skipping duplicate dispatch"
                    )
                    continue

                effective_sample_rate = capture_sample_rate or 48000
                audio_duration_ms = int(len(audio_data) / (effective_sample_rate * 2) * 1000)
                speech_end_time = time.time()
                total_utterance_duration = (
                    (speech_end_time - speech_start_time) if speech_start_time else audio_duration_ms / 1000.0
                )

                current_utterance_id = expected_id or str(uuid.uuid4())
                utterance_in_flight = True

                now_ts = time.time()
                if now_ts - self._last_finalize_ts < 0.75:
                    logger.info(
                        f"🛑 Debounced duplicate end-of-utterance (ID={current_utterance_id}); skipping finalize within 750ms window"
                    )
                    buffer.clear()
                    silence_frames = 0
                    consecutive_start_frames = 0
                    utterance_in_flight = False
                    continue
                self._last_finalize_ts = now_ts

                logger.info("")
                logger.info("═══════════════════════════════════════════════════════════")
                logger.info("🔊 SINGLE STT CALL - Complete user turn captured")
                logger.info(f"   Utterance ID: {current_utterance_id}")
                logger.info(f"   Audio buffer: {len(audio_data)} bytes = {audio_duration_ms}ms")
                logger.info(f"   Total duration: {total_utterance_duration:.1f}s (from speech start to now)")
                logger.info(f"   Finalization trigger: {SILENCE_DURATION_MS}ms continuous silence")
                logger.info("═══════════════════════════════════════════════════════════")
                logger.info("")

                try:
                    await self._save_utterance_audio(
                        audio_bytes=audio_data,
                        utterance_id=current_utterance_id,
                        duration_ms=audio_duration_ms,
                        sample_rate=effective_sample_rate,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save utterance audio ({current_utterance_id}): {e}")

                try:
                    prepared_bytes, prep_metrics = self._prepare_audio_for_stt(audio_data, effective_sample_rate)
                    logger.info(
                        f"🎧 STT prep metrics: dur={prep_metrics['duration_ms']:.0f}ms "
                        f"rms={prep_metrics['rms']:.4f} snr={prep_metrics['snr_db']:.1f}dB peak={prep_metrics['peak']:.3f}"
                    )

                    try:
                        await self._save_conditioned_audio(
                            audio_bytes=prepared_bytes,
                            utterance_id=current_utterance_id,
                            duration_ms=int(prep_metrics["duration_ms"]),
                            sample_rate=48000,
                        )
                    except Exception:
                        pass

                    # Call STT provider
                    transcript = await self.orchestrator.stt.transcribe(prepared_bytes)
                    logger.info(f"✅ STT returned ({len(transcript)} chars, utterance_id={current_utterance_id}): '{transcript}'")

                    # CRITICAL: Check for stop-words BEFORE applying quality filters
                    # Stop-words can be ultra-short (<400ms) and still valid
                    is_likely_stop_word = any(
                        keyword in transcript.lower() 
                        for keyword in ["stop", "wait", "hold", "pause", "quit", "exit"]
                    )
                    
                    if (
                        prep_metrics["duration_ms"] < STT_MIN_DURATION_MS
                        or prep_metrics["rms"] < STT_MIN_RMS
                        or prep_metrics["snr_db"] < STT_MIN_SNR_DB
                    ):
                        # BYPASS quality filters if transcript contains stop-words
                        if is_likely_stop_word:
                            logger.info(
                                f"✅ [stop-word-bypass] Audio below quality threshold BUT contains stop-word ('{transcript}'), processing anyway"
                            )
                        else:
                            logger.warning(
                                "🛑 Prepared audio below floor (dur/rms/snr). Skipping this utterance silently - user will naturally rephrase."
                            )
                            # Don't send "I couldn't hear" message - just skip and wait for clearer audio
                            # User's natural flow: speak again louder/clearer without AI interrupting
                            utterance_in_flight = False
                            await self.controller._set_waiting_for_user()
                            continue

                    transcript = await self.orchestrator.stt.transcribe(prepared_bytes)
                    logger.info(
                        f"✅ STT returned ({len(transcript)} chars, utterance_id={current_utterance_id}): '{transcript}'"
                    )

                except Exception as e:
                    logger.error(
                        f"❌ STT or preparation failed for utterance_id={current_utterance_id}: {e}",
                        exc_info=True,
                    )
                    transcript = ""
                finally:
                    utterance_in_flight = False

                if not transcript.strip():
                    logger.warning(f"⚠️ Empty transcript for utterance_id={current_utterance_id}, skipping")
                    continue

                # Stop-word guard even when AI has finished speaking; prevents control phrases from hitting LLM
                try:
                    match = await self.stop_interruptor.check_for_stop_words(
                        transcript=transcript,
                        buffer_duration_ms=int(prep_metrics.get("duration_ms", 0)),
                        stt_confidence=0.9,
                        time_since_ai_start_ms=0,
                        utterance_id=current_utterance_id,
                    )
                    logger.info(
                        "🛑 [stop-final] Transcript check | text='%s' match=%s tier=%s",
                        transcript[:80],
                        match.matched,
                        getattr(match, "tier", None),
                    )
                    if match.matched and match.tier == InterruptionTier.TIER_1_HARD:
                        logger.info(
                            f"🛑 TIER-1 STOP-WORD DETECTED: '{match.keyword}' | "
                            f"This is a control command, NOT user speech - routing to USER_OVERRIDE"
                        )
                        await self._notify(
                            {
                                "type": "stop_word",
                                "keyword": match.keyword or transcript,
                                "utteranceId": current_utterance_id,
                                "roomId": self.room_id,
                                "timestamp": time.time(),
                            }
                        )
                        if hasattr(self, "controller") and self.controller:
                            await self.controller._notify_state(ConversationPhase.USER_OVERRIDE)
                            await self.controller._set_waiting_for_user()
                        # Skip normal processing to keep stop command out of LLM flow
                        utterance_in_flight = False  # Reset flag before continue
                        continue
                except Exception as e:
                    logger.warning(f"[stop-final] stop word detection failed: {e}")

                await self._notify(
                    {
                        "type": "transcript",
                        "role": "user",
                        "text": transcript,
                        "utteranceId": current_utterance_id,
                        "roomId": self.room_id,
                        "timestamp": time.time(),
                    }
                )
                
                # Emit USER_SPEAKING_END after transcript is ready
                await self._notify({
                    'type': 'speaking_state',
                    'speaker': 'user',
                    'state': 'end',
                    'utteranceId': current_utterance_id,
                    'roomId': self.room_id,
                    'timestamp': time.time(),
                })

                if hasattr(self, "controller") and self.controller:
                    logger.info(
                        f"🎮 Controller exists, calling handle_user_utterance with utterance_id={current_utterance_id}..."
                    )
                    try:
                        # NEW: Check if this is a barge-in interruption
                        if current_phase == ConversationPhase.AI_SPEAKING and barge_in_decision == "interrupt_ai":
                            logger.info("🛑 INTERRUPTION ROUTE: User spoke during AI - canceling TTS immediately")
                            await self.cancel_tts()
                            # Route to interruption handler (will transition AI_SPEAKING → PROCESSING_USER)
                            await self.controller.handle_interruption(transcript, current_utterance_id)
                        else:
                            # Normal route: User spoke during WAIT_FOR_USER
                            async with self._utterance_processing_lock:
                                await self.controller.handle_user_utterance(transcript, current_utterance_id)
                                self._last_consumed_utterance_id = current_utterance_id
                        logger.info(f"✅ Controller processing completed for utterance_id={current_utterance_id}")
                    except Exception as e:
                        logger.error(
                            f"❌ Controller handling failed for utterance_id={current_utterance_id}: {e}",
                            exc_info=True,
                        )
                else:
                    logger.warning("⚠️ No controller found, skipping processing")
                    persona_slug = "default"
                    try:
                        from ai_agent.models import Call

                        @database_sync_to_async
                        def get_fallback_persona():
                            call = Call.objects.filter(room__id=self.room_id).order_by("-started_at").first()
                            if call and call.persona:
                                return call.persona.slug
                            return "default"

                        persona_slug = await get_fallback_persona()
                    except Exception:
                        pass

                    result = await self.orchestrator.handle_utterance(self.room_id, audio_data, persona_slug)
                    if result and result.get("tts_audio"):
                        await self.queue_audio_output(result["tts_audio"])
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")

    async def _save_utterance_audio(self, audio_bytes: bytes, utterance_id: str, duration_ms: int, sample_rate: int = 48000):
        """Save the exact audio buffer being sent to STT for offline testing.

        - Saves raw PCM (s16le mono) for exact byte-level match with STT input
        - Also saves a WAV copy (48kHz mono) for easy playback/inspection
        - Files are stored under backend/test_recordings/{roomId}/
        """
        try:
            import os
            import io
            import wave
            from pathlib import Path

            # Allow disabling via env flag if desired
            record_flag = os.environ.get("RECORD_UTTERANCES", "true").lower() in {"1", "true", "yes", "on"}
            if not record_flag:
                return

            backend_root = Path(__file__).resolve().parents[1]  # .../backend
            out_dir = backend_root / "test_recordings" / str(self.room_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = int(time.time() * 1000)
            base_name = f"{ts}__{utterance_id}__{duration_ms}ms"

            # 1) Save raw PCM exactly as sent to STT
            pcm_path = out_dir / f"{base_name}.pcm"
            async def _write_pcm():
                with open(pcm_path, "wb") as f:
                    f.write(audio_bytes)

            # 2) Save WAV copy for convenience
            wav_path = out_dir / f"{base_name}.wav"
            async def _write_wav():
                with io.BytesIO() as bio:
                    with wave.open(bio, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(sample_rate)
                        wav.writeframes(audio_bytes)
                    data = bio.getvalue()
                with open(wav_path, "wb") as f:
                    f.write(data)

            # Perform writes in thread to avoid blocking event loop
            await asyncio.gather(
                asyncio.to_thread(lambda: asyncio.run(_write_pcm())),
                asyncio.to_thread(lambda: asyncio.run(_write_wav()))
            )

            logger.info(f"💾 Saved utterance audio to: {pcm_path} and {wav_path}")
        except Exception as e:
            logger.warning(f"Audio save failed: {e}")

    async def _save_conditioned_audio(self, audio_bytes: bytes, utterance_id: str, duration_ms: int, sample_rate: int = 48000):
        """Save conditioned audio buffer as WAV for inspection alongside raw.

        Files are stored under backend/test_recordings/{roomId}/ and suffixed with '__cond.wav'.
        """
        try:
            import io
            import wave
            from pathlib import Path

            backend_root = Path(__file__).resolve().parents[1]
            out_dir = backend_root / "test_recordings" / str(self.room_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = int(time.time() * 1000)
            wav_path = out_dir / f"{ts}__{utterance_id}__{duration_ms}ms__cond.wav"

            with io.BytesIO() as bio:
                with wave.open(bio, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_bytes)
                data = bio.getvalue()

            with open(wav_path, "wb") as f:
                f.write(data)

            logger.info(f"💾 Saved conditioned audio to: {wav_path}")
        except Exception as e:
            logger.debug(f"Conditioned audio save failed: {e}")

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
        # Ignore barge-in during initial playback ramp
        self._ai_playback_guard_deadline = time.time() + 0.4
        self._ai_playback_started_at = time.time()
        # Reset cancel flag
        self._tts_cancel_event.clear()
        # Reset barge-in machine to initial state (ready for detection)
        if self.barge_in_machine:
            self.barge_in_machine.reset()
        self._stopword_buffer.clear()
        # Notify UI that AI speaking is starting (explicit state event)
        try:
            await self._notify({
                'type': 'speaking_state',
                'speaker': 'ai',
                'state': 'start',
                'turnId': turn_id,
                'roomId': self.room_id,
                'timestamp': time.time(),
            })
            # Also send legacy ai_playback for backward compatibility
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
                if self._tts_cancel_event.is_set():
                    logger.info("⏹️  TTS playback canceled mid-stream; stopping queueing further chunks")
                    break
                chunk = audio_bytes[i:i+CHUNK_SIZE]
                if chunk:  # Only queue non-empty chunks
                    await self.ai_track.q.put(chunk)
                    chunks_queued += 1
            
            logger.info(f"🔊 Queued {chunks_queued} chunks ({duration_seconds:.2f}s of audio) - VAD disabled to prevent echo")
            
            # CRITICAL: Wait for audio to finish OR cancel event, before re-enabling VAD
            # Add 15% buffer to ensure playback completes + any codec/network jitter
            wait_time = duration_seconds * 1.15
            try:
                await asyncio.wait_for(self._wait_or_cancel(wait_time), timeout=wait_time + 0.1)
            except asyncio.TimeoutError:
                pass
            
        finally:
            # ALWAYS re-enable VAD, even if there was an error
            self.ai_is_speaking = False
            logger.info(f"🎤 AI finished speaking - VAD will re-enable after grace period")
            self._ai_playback_started_at = None
            # Notify UI that AI speaking has ended (explicit state event)
            try:
                await self._notify({
                    'type': 'speaking_state',
                    'speaker': 'ai',
                    'state': 'end',
                    'turnId': turn_id,
                    'canceled': self._tts_cancel_event.is_set(),
                    'roomId': self.room_id,
                    'timestamp': time.time(),
                })
                # Also send legacy ai_playback for backward compatibility
                if turn_id:
                    await self._notify({
                        'type': 'ai_playback',
                        'status': 'canceled' if self._tts_cancel_event.is_set() else 'end',
                        'turnId': turn_id,
                        'roomId': self.room_id,
                        'timestamp': time.time(),
                    })
            except Exception:
                pass

    async def _wait_or_cancel(self, seconds: float):
        """Wait for given seconds or until cancel is signaled, whichever comes first."""
        try:
            await asyncio.wait_for(self._tts_cancel_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def cancel_tts(self):
        """Signal cancellation and drain any queued audio to stop playback quickly."""
        try:
            if not self.ai_is_speaking:
                return
            self._tts_cancel_event.set()
            # Drain queue non-blocking
            drained = 0
            while not self.ai_track.q.empty():
                try:
                    _ = self.ai_track.q.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            logger.info(f"🧹 Drained {drained} queued audio chunks after cancel")
        except Exception as e:
            logger.debug(f"cancel_tts error: {e}")

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

    def _prepare_audio_for_stt(self, audio_bytes: bytes, sample_rate: int = 48000):
        """Trim edges, normalize loudness, and compute simple metrics for STT input."""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return audio_bytes, {"duration_ms": 0.0, "rms": 0.0, "snr_db": -60.0, "peak": 0.0}

        samples /= 32767.0

        window = max(int(0.02 * sample_rate), 1)
        abs_samples = np.abs(samples)
        noise_est = float(np.median(abs_samples))
        trim_threshold = max(noise_est * 3.0, 0.003)

        start_idx = 0
        for i in range(0, len(samples), window):
            if np.sqrt(np.mean(abs_samples[i:i + window] ** 2)) > trim_threshold:
                start_idx = max(0, i - window)
                break

        end_idx = len(samples)
        for i in range(len(samples) - window, 0, -window):
            if np.sqrt(np.mean(abs_samples[i:i + window] ** 2)) > trim_threshold:
                end_idx = min(len(samples), i + 2 * window)
                break

        trimmed = samples[start_idx:end_idx] if end_idx > start_idx else samples

        rms = float(np.sqrt(np.mean(trimmed ** 2))) if trimmed.size else 0.0
        peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
        snr_db = 20 * np.log10(max(rms, 1e-6) / max(noise_est, 1e-6)) if trimmed.size else -60.0

        target_rms = 0.1
        if rms > 0:
            gain = min(4.0, target_rms / max(rms, 1e-6))
            trimmed = np.clip(trimmed * gain, -0.97, 0.97)
            rms = float(np.sqrt(np.mean(trimmed ** 2)))
            peak = float(np.max(np.abs(trimmed)))

        duration_ms = len(trimmed) / sample_rate * 1000 if sample_rate else 0.0
        prepared_bytes = (trimmed * 32767.0).astype(np.int16).tobytes()

        return prepared_bytes, {
            "duration_ms": duration_ms,
            "rms": rms,
            "snr_db": snr_db,
            "peak": peak,
        }

    def _boost_audio_for_stt(self, audio_bytes: bytes, gain: float = 2.0) -> bytes:
        """Apply a simple gain to PCM audio for retrying STT on low-volume buffers."""
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            samples *= gain
            np.clip(samples, -32767.0, 32767.0, out=samples)
            return samples.astype(np.int16).tobytes()
        except Exception:
            return audio_bytes

    async def _notify(self, message: dict):
        """Send a UI event if a notify callback is available."""
        try:
            if self.notify_callback:
                await self.notify_callback(message)
        except Exception as e:
            logger.debug(f"Notify callback failed: {e}")


