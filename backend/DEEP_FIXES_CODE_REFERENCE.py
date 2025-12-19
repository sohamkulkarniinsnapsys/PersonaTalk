"""
DEEP FIXES QUICK REFERENCE
Quick guide to the 6 critical architectural fixes

All code patterns are now in:
- backend/ai_agent/webrtc.py (audio processing)
- backend/ai_agent/conversation.py (state machine)
"""

# ============================================================================
# FIX #1: VAD STABILITY WINDOW (webrtc.py, lines 251-370)
# ============================================================================
# Problem: Early utterance cutoff during natural pauses
# Solution: Three-condition end-of-utterance with stability window

# Key variables:
MIN_BUFFER_MS = 1500  # Don't end utterance until buffer >= 1.5s
SILENCE_DURATION_MS = 1200  # Must have 1.2s silence
STABILITY_WINDOW_MS = 400  # After silence, wait 400ms for speech onset

# Key pattern:
if silence_frames >= SILENCE_FRAME_COUNT:
    # CONDITION 1: We have enough silence
    # Check CONDITION 2: buffer size before entering stability window
    if stability_countdown == 0 and len(buffer) >= MIN_BUFFER_BYTES:
        # Buffer is large enough; now enter stability phase
        stability_countdown = STABILITY_WINDOW_FRAMES
        logger.info(f"⏱️ Entering stability window")
        continue
    elif stability_countdown > 0:
        # In stability window: if ANY speech onset happens, extend utterance
        if energy > START_THRESHOLD:
            logger.info(f"🔄 Speech during stability window; extending utterance")
            stability_countdown = 0
            continue  # Keep accumulating
        else:
            # Still in stability window, no speech; decrement
            stability_countdown -= 1
            if stability_countdown <= 0:
                # CONDITION 3: Stability window expired
                logger.info(f"✅ Stability window expired; utterance complete")
                break  # Process utterance


# ============================================================================
# FIX #2: UTTERANCE ID BINDING (webrtc.py, lines 320-330, 555-595)
# ============================================================================
# Problem: Race conditions where expected_id changes before STT returns
# Solution: Bind utterance_id to audio buffer at capture time

# Key variables:
current_utterance_id = None  # Bound at buffer creation
current_buffer_audio = None  # Reference to audio data
utterance_in_flight = False  # STT status flag

# Key pattern (at utterance end detection):
# CRITICAL: Bind utterance_id to THIS audio buffer BEFORE dispatching STT
current_utterance_id = expected_id or str(uuid.uuid4())
current_buffer_audio = audio_data
utterance_in_flight = True

logger.info(f"🔊 Calling STT for {len(audio_data)} bytes (utterance_id={current_utterance_id})...")

try:
    transcript = await self.orchestrator.stt.transcribe(audio_data)
finally:
    # ALWAYS reset in-flight flag after STT (even on error)
    utterance_in_flight = False

# Later in controller processing:
utterance_id = current_utterance_id  # USE THE BOUND ID
# NOT the current expected_id (which may have advanced to utt-2)


# ============================================================================
# FIX #3: DUPLICATE STT PREVENTION (webrtc.py, lines 477-495)
# ============================================================================
# Problem: Same audio buffer sent to STT twice
# Solution: utterance_in_flight flag blocks duplicates

# Key pattern:
if utterance_in_flight:
    logger.warning(f"⚠️ STT already in flight; skipping duplicate")
    continue

# Set flag BEFORE dispatching
utterance_in_flight = True

# Reset flag AFTER STT returns
try:
    transcript = await self.orchestrator.stt.transcribe(audio_data)
finally:
    utterance_in_flight = False


# ============================================================================
# FIX #4: LOW-INFO CLARIFICATION (conversation.py, lines 280-310)
# ============================================================================
# Problem: Low-info utterances silently dropped
# Solution: Ask for clarification with spoken prompt

# Key pattern:
if not self._is_meaningful(text):
    logger.info(f"⏸️ Low-information utterance detected: '{text}'")
    await self._notify_state(ConversationPhase.CLARIFICATION_REQUIRED)
    
    # Speak to user
    clarification_prompt = "I didn't quite get that. Could you give me a bit more detail?"
    audio_bytes = await self.orchestrator.tts.synthesize(
        clarification_prompt,
        self.session.persona_config.get('voice', {})
    )
    await self.send_audio(audio_bytes, self.session.last_ai_turn_id)
    
    # Stay in WAIT_FOR_USER; do NOT advance
    self.session.active_user_utterance_id = None
    await self._set_waiting_for_user()
    return


# ============================================================================
# FIX #5: INCOMPLETE UTTERANCE DETECTION (conversation.py, lines 797-870)
# ============================================================================
# Problem: LLM called on incomplete thoughts ("I remember a difference between")
# Solution: Detect incomplete grammar and wait for continuation

# Key method:
def _has_incomplete_structure(self, text: str) -> bool:
    """Detect utterances ending with filler/conjunction/preposition."""
    
    incomplete_endings = [
        "between",   # "difference between" (waiting for two things)
        "and",       # "X and" (waiting for what comes after)
        "or",        # "X or" (waiting for alternatives)
        "is",        # "is..." (waiting for predicate)
        # ... etc
    ]
    
    s = text.strip().lower()
    for ending in incomplete_endings:
        if s.endswith(ending):
            logger.debug(f"📍 Incomplete structure: ends with '{ending}'")
            return True
    
    return False

# Key pattern in handle_user_utterance:
await self._notify_state(ConversationPhase.VALIDATING_UTTERANCE)
if self._has_incomplete_structure(text):
    logger.info(f"🔤 Utterance incomplete; waiting for continuation")
    # DON'T call LLM
    # Stay in WAIT_FOR_USER
    self.session.active_user_utterance_id = None
    await self._set_waiting_for_user()
    return

# User's next utterance will auto-merge with history


# ============================================================================
# FIX #6: BARGE-IN SUPPORT (webrtc.py, lines 375-420)
# ============================================================================
# Problem: TTS blocks user speech; no interruption possible
# Solution: Detect user speech during AI playback and cancel immediately

# Key pattern (inside VAD loop):
if self.ai_is_speaking:
    # Detect energy WHILE AI is speaking
    if energy > START_THRESHOLD:
        consecutive_voiced += 1
    else:
        consecutive_voiced = 0
    
    # Barge-in detected: user speaking >= 60ms
    if consecutive_voiced >= MIN_CONSECUTIVE_VOICED:
        logger.info(f"🛑 BARGE-IN DETECTED - Canceling TTS immediately")
        await self.cancel_tts()  # Stops TTS playback
        
        # CRITICAL: Reset grace period to 0
        # Allows immediate capture of user's speech
        post_ai_grace_frames = 0
        
        # Reset VAD state for fresh utterance
        speaking = False
        buffer = bytearray()
        silence_frames = 0
        consecutive_voiced = 0
        
        # Emit event to UI
        await self._notify({
            'type': 'barge_in',
            'status': 'detected',
            'roomId': self.room_id,
            'timestamp': time.time(),
        })
        
        # Fall through to normal VAD (ai_is_speaking now False)
    else:
        # Still accumulating voice frames
        logger.debug(f"VAD skipped (AI speaking) - energy={energy}")
        continue

# cancel_tts() implementation:
async def cancel_tts(self):
    """Signal cancellation and drain queued audio."""
    if not self.ai_is_speaking:
        return
    self._tts_cancel_event.set()  # Signal cancellation
    
    # Drain queue non-blocking
    drained = 0
    while not self.ai_track.q.empty():
        try:
            _ = self.ai_track.q.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            break
    
    logger.info(f"🧹 Drained {drained} chunks; TTS stopped")


# ============================================================================
# TESTING CHECKLIST
# ============================================================================

# [] VAD Stability: Say sentence with pause → should NOT cut off
# [] Min Buffer: Speak < 1.5s → should get clarification "more detail?"
# [] No Duplicates: Monitor logs → one "🔊 Calling STT" per utterance
# [] Incomplete: Say "I like" pause → say "useEffect" → one turn
# [] Clarification: Speak quietly → hear "didn't quite get that"
# [] Barge-in: Speak while AI talking → TTS stops within 1 frame
# [] Logs: grep for "🛑 BARGE-IN" (should see when interrupting)
# [] Logs: grep for "Stale utterance_id" (should see NONE)

