"""
Explicit state machine for intelligent barge-in handling.

This module defines the states and transitions for the two-stage interruption pipeline:
1. Speech Validation: Energy-based detection (proven reliable)
2. Duration Gate: Speech must be sustained for minimum duration
3. Query Validation: Verify speech is a meaningful interruption attempt (ASR confirmation)

States enforce clean transitions and prevent race conditions during AI playback.

CRITICAL: Uses energy/SNR-based detection (Silero VAD removed due to frame size incompatibility).
"""

import enum
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BargeinState(str, enum.Enum):
    """States in the barge-in detection pipeline.
    
    State Flow:
        AI_SPEAKING → (detect energy) → BARGE_IN_CANDIDATE → (speech validation) → 
        BUFFERING_QUERY → (query validation) → INTERRUPTION_ACCEPTED → USER_SPEAKING
        
        OR at any stage → INTERRUPTION_REJECTED → AI_SPEAKING (continue playback)
    """
    # AI is actively playing TTS output
    AI_SPEAKING = "AI_SPEAKING"
    
    # Detected potential speech, accumulating consecutive voiced frames
    BARGE_IN_CANDIDATE = "BARGE_IN_CANDIDATE"
    
    # Speech validation passed, now buffering audio for query validation
    BUFFERING_QUERY = "BUFFERING_QUERY"
    
    # Both validations passed, interruption accepted
    INTERRUPTION_ACCEPTED = "INTERRUPTION_ACCEPTED"
    
    # Validation failed, rejected as noise/accidental
    INTERRUPTION_REJECTED = "INTERRUPTION_REJECTED"
    
    # User is speaking (normal state, not during AI playback)
    USER_SPEAKING = "USER_SPEAKING"
    
    # User finished speaking, processing utterance
    PROCESSING_UTTERANCE = "PROCESSING_UTTERANCE"


@dataclass
class BargeinContext:
    """Runtime context for tracking barge-in detection state.
    
    This is mutable state that gets updated frame-by-frame during audio processing.
    Reset when transitioning between major states (AI_SPEAKING ↔ USER_SPEAKING).
    """
    current_state: BargeinState = BargeinState.AI_SPEAKING
    
    # Speech Validation State (VAD-based)
    consecutive_speech_frames: int = 0  # VAD-classified speech frames
    speech_validation_start_time: Optional[float] = None
    validation_buffer: bytearray = field(default_factory=bytearray)
    vad_speech_probability: float = 0.0  # Latest VAD probability
    
    # Query Validation State
    query_buffer: bytearray = field(default_factory=bytearray)
    query_buffer_start_time: Optional[float] = None
    preliminary_transcript: Optional[str] = None
    
    # Energy Tracking (kept for logging, but not used for detection)
    peak_energy_during_validation: int = 0
    avg_snr_during_validation: float = 0.0
    
    # Timing
    state_entry_time: float = field(default_factory=time.time)
    last_transition_time: float = field(default_factory=time.time)
    
    # Decision Metadata
    rejection_reason: Optional[str] = None
    acceptance_reason: Optional[str] = None
    
    def reset(self):
        """Reset all state for fresh detection cycle."""
        self.current_state = BargeinState.AI_SPEAKING
        self.consecutive_speech_frames = 0
        self.speech_validation_start_time = None
        self.validation_buffer.clear()
        self.query_buffer.clear()
        self.query_buffer_start_time = None
        self.preliminary_transcript = None
        self.vad_speech_probability = 0.0
        self.peak_energy_during_validation = 0
        self.avg_snr_during_validation = 0.0
        self.rejection_reason = None
        self.acceptance_reason = None
        self.state_entry_time = time.time()
        self.last_transition_time = time.time()
    
    def transition_to(self, new_state: BargeinState, reason: str = ""):
        """Explicit state transition with logging."""
        old_state = self.current_state
        self.current_state = new_state
        self.last_transition_time = time.time()
        
        logger.info(f"🔄 BARGE-IN STATE TRANSITION: {old_state.value} → {new_state.value}")
        if reason:
            logger.info(f"   Reason: {reason}")
        
        # Log context when transitioning to terminal states
        if new_state == BargeinState.INTERRUPTION_ACCEPTED:
            logger.info(f"   ✅ Accepted: {self.acceptance_reason or 'No reason provided'}")
            logger.info(f"   Peak energy: {self.peak_energy_during_validation}")
            logger.info(f"   Avg SNR: {self.avg_snr_during_validation:.1f} dB")
            logger.info(f"   Validation buffer: {len(self.validation_buffer)} bytes")
            logger.info(f"   Query buffer: {len(self.query_buffer)} bytes")
        elif new_state == BargeinState.INTERRUPTION_REJECTED:
            logger.info(f"   ❌ Rejected: {self.rejection_reason or 'No reason provided'}")
            logger.info(f"   Peak energy: {self.peak_energy_during_validation}")
            logger.info(f"   Avg SNR: {self.avg_snr_during_validation:.1f} dB")
            logger.info(f"   Consecutive frames: {self.consecutive_speech_frames}")


class BargeinStateMachine:
    """Manages state transitions for intelligent barge-in detection.
    
    This class encapsulates all state transition logic and validation rules,
    keeping webrtc.py focused on audio processing.
    
    CRITICAL: Now uses Silero VAD for speech classification instead of energy thresholds.
    """
    
    def __init__(self, behavior_config, vad_validator=None):
        """Initialize state machine with persona-specific behavior.
        
        Args:
            behavior_config: BargeinBehavior instance with thresholds and rules
            vad_validator: DEPRECATED - no longer used (causes frame size errors)
        """
        self.behavior = behavior_config
        self.context = BargeinContext()
        
        # NO VAD: Energy-based detection is more reliable for frame-by-frame processing
        # Silero VAD requires minimum chunk sizes (512 samples) that don't align with 20ms frames
        # Instead, use proven energy/SNR thresholds that have been tested extensively
        self.vad_validator = None
        
        logger.info(f"🎮 Initialized BargeinStateMachine for persona: {behavior_config.persona_type}")
        logger.info(f"   Using energy-based speech detection (Silero VAD disabled)")
        logger.info(f"   Min energy: {behavior_config.min_energy_threshold}, Min SNR: {behavior_config.min_snr_db} dB")
    
    def reset(self):
        """Reset to initial state (called when AI starts speaking)."""
        self.context.reset()
        if self.vad_validator:
            self.vad_validator.reset()
        logger.info("🔄 BargeinStateMachine reset to initial state")
    
    def process_frame(self, energy_peak: int, snr_db: float, raw_bytes: bytes, 
                     noise_floor: float) -> Dict[str, Any]:
        """Process a single audio frame and update state.
        
        Args:
            energy_peak: Peak amplitude in frame
            snr_db: Signal-to-Noise Ratio in decibels
            raw_bytes: Raw audio data (s16 mono)
            noise_floor: Current noise floor estimate
            
        Returns:
            Dict with keys:
                - action: "continue_ai" | "accept_barge_in" | "reject_barge_in"
                - new_state: Updated BargeinState
                - reason: Human-readable decision reason
        """
        state = self.context.current_state
        
        # Update running statistics
        if energy_peak > self.context.peak_energy_during_validation:
            self.context.peak_energy_during_validation = energy_peak
        
        # Calculate dynamic threshold based on noise floor
        dynamic_threshold = max(
            self.behavior.min_energy_threshold,
            noise_floor * 32767 * 8  # reduced multiplier for more sensitivity
        )
        
        # State-specific processing (energy-based ONLY)
        if state == BargeinState.AI_SPEAKING:
            return self._handle_ai_speaking(energy_peak, snr_db, raw_bytes, dynamic_threshold)
        
        elif state == BargeinState.BARGE_IN_CANDIDATE:
            return self._handle_barge_candidate(energy_peak, snr_db, raw_bytes, dynamic_threshold)
        
        elif state == BargeinState.BUFFERING_QUERY:
            return self._handle_buffering_query(energy_peak, snr_db, raw_bytes)
        
        else:
            # In terminal states (ACCEPTED/REJECTED/USER_SPEAKING), no further processing
            return {
                "action": "continue_ai" if state == BargeinState.AI_SPEAKING else "no_action",
                "new_state": state,
                "reason": f"In terminal state {state.value}"
            }
    
    def _handle_ai_speaking(self, energy_peak: int, snr_db: float, 
                           raw_bytes: bytes, threshold: int) -> Dict[str, Any]:
        """Handle frame while AI is speaking normally.
        
        Uses energy-based detection (proven reliable, no frame size issues).
        """
        logger.debug(f"Energy-based detection: peak={energy_peak}, threshold={threshold}, snr={snr_db:.1f}")
        
        if energy_peak > threshold and snr_db >= self.behavior.min_snr_db:
            self.context.consecutive_speech_frames += 1
            
            # Log progress every 10 frames to avoid spam
            if self.context.consecutive_speech_frames % 10 == 0:
                logger.info(
                    f"📊 Barge-in candidacy: {self.context.consecutive_speech_frames} frames "
                    f"(need {self.behavior.min_consecutive_frames}) | energy={energy_peak}, snr={snr_db:.1f} dB"
                )
            
            # Transition to CANDIDATE state if sustained
            if self.context.consecutive_speech_frames >= self.behavior.min_consecutive_frames:
                self.context.transition_to(
                    BargeinState.BARGE_IN_CANDIDATE,
                    f"Energy-based: sustained speech detected ({self.context.consecutive_speech_frames} frames = {self.context.consecutive_speech_frames * 20}ms)"
                )
                self.context.speech_validation_start_time = time.time()
                self.context.validation_buffer = bytearray(raw_bytes)
                self.context.avg_snr_during_validation = snr_db
                
                logger.info(f"✅ Entered BARGE_IN_CANDIDATE state (sustained speech detected)")
                
                return {
                    "action": "continue_ai",  # Still validating, don't cancel yet
                    "new_state": BargeinState.BARGE_IN_CANDIDATE,
                    "reason": f"Energy-based: sustained speech ({self.context.consecutive_speech_frames * 20}ms)"
                }
        else:
            # Reset counter if energy drops
            if self.context.consecutive_speech_frames > 0:
                logger.debug(
                    f"⏮️  Barge-in reset: energy {energy_peak} < {threshold} or "
                    f"SNR {snr_db:.1f} < {self.behavior.min_snr_db}"
                )
            self.context.consecutive_speech_frames = 0
        
        return {
            "action": "continue_ai",
            "new_state": BargeinState.AI_SPEAKING,
            "reason": f"No barge-in yet (energy={energy_peak}, snr={snr_db:.1f}, need energy>{threshold} & snr>{self.behavior.min_snr_db})"
        }
    
    def _handle_barge_candidate(self, energy_peak: int, snr_db: float,
                               raw_bytes: bytes, threshold: int) -> Dict[str, Any]:
        """Handle frame during speech validation phase.
        
        Uses energy-based detection exclusively (VAD removed due to frame size incompatibility).
        Accumulates audio for query validation while confirming sustained speech.
        """
        # Continue accumulating buffer
        self.context.validation_buffer.extend(raw_bytes)
        
        # Update SNR running average (kept for logging)
        frame_count = len(self.context.validation_buffer) // (960 * 2)  # 960 samples/frame
        self.context.avg_snr_during_validation = (
            (self.context.avg_snr_during_validation * max(0, frame_count - 1) + snr_db) / max(1, frame_count)
        )
        
        # Energy-based continuation: Confirm sustained speech
        if energy_peak > threshold and snr_db >= self.behavior.min_snr_db:
            self.context.consecutive_speech_frames += 1
            
            # Check if validation duration met
            elapsed_ms = (time.time() - self.context.speech_validation_start_time) * 1000
            
            if elapsed_ms % 100 < 20:  # Log every ~100ms
                logger.info(
                    f"🎙️ Speech validation in progress: {elapsed_ms:.0f}ms / "
                    f"{self.behavior.speech_validation_duration_ms}ms, "
                    f"buffer={len(self.context.validation_buffer)} bytes"
                )
            
            if elapsed_ms >= self.behavior.speech_validation_duration_ms:
                # Speech validation PASSED → Move to query validation
                self.context.transition_to(
                    BargeinState.BUFFERING_QUERY,
                    f"Energy-based: Speech sustained for {elapsed_ms:.0f}ms"
                )
                self.context.query_buffer = bytearray(self.context.validation_buffer)
                self.context.query_buffer_start_time = time.time()
                
                logger.info(f"✅ Speech validation PASSED ({elapsed_ms:.0f}ms, moving to query validation)")
                
                return {
                    "action": "continue_ai",  # Still validating query
                    "new_state": BargeinState.BUFFERING_QUERY,
                    "reason": f"Speech validation passed, buffering query ({elapsed_ms:.0f}ms)"
                }
        else:
            # Energy dropped → REJECT
            self.context.rejection_reason = (
                f"Energy-based: Speech validation failed (energy dropped to {energy_peak}, "
                f"threshold {threshold}, SNR {snr_db:.1f} dB)"
            )
            self.context.transition_to(BargeinState.INTERRUPTION_REJECTED, self.context.rejection_reason)
            
            logger.info(f"❌ Speech validation REJECTED: {self.context.rejection_reason}")
            
            self.context.reset()
            
            return {
                "action": "reject_barge_in",
                "new_state": BargeinState.AI_SPEAKING,
                "reason": self.context.rejection_reason
            }
        
        return {
            "action": "continue_ai",
            "new_state": BargeinState.BARGE_IN_CANDIDATE,
            "reason": f"Speech validation in progress ({len(self.context.validation_buffer)} bytes)"
        }
    
    def _handle_buffering_query(self, energy_peak: int, snr_db: float,
                               raw_bytes: bytes) -> Dict[str, Any]:
        """Handle frame during query validation buffering."""
        # Continue accumulating buffer
        self.context.query_buffer.extend(raw_bytes)
        
        # Check if buffer duration met
        elapsed_ms = (time.time() - self.context.query_buffer_start_time) * 1000
        
        if elapsed_ms >= self.behavior.buffer_duration_ms:
            # Buffer complete → Run query validation (will be done by caller)
            # For now, transition to ACCEPTED (caller will validate)
            self.context.acceptance_reason = (
                f"Query buffer complete ({elapsed_ms:.0f}ms, {len(self.context.query_buffer)} bytes)"
            )
            self.context.transition_to(BargeinState.INTERRUPTION_ACCEPTED, self.context.acceptance_reason)
            
            return {
                "action": "accept_barge_in",
                "new_state": BargeinState.INTERRUPTION_ACCEPTED,
                "reason": self.context.acceptance_reason,
                "query_buffer": bytes(self.context.query_buffer)
            }
        
        return {
            "action": "continue_ai",
            "new_state": BargeinState.BUFFERING_QUERY,
            "reason": f"Buffering query ({elapsed_ms:.0f}/{self.behavior.buffer_duration_ms}ms)"
        }
    
    async def validate_query(self, audio_buffer: bytes, stt_provider) -> bool:
        """Run Stage 2 validation: Is this a meaningful query?
        
        Args:
            audio_buffer: Audio buffer to transcribe
            stt_provider: STT provider instance
            
        Returns:
            True if query is valid (barge-in should be accepted)
            False if query is invalid (reject barge-in)
        """
        try:
            # Perform quick STT on buffered audio
            transcript = await stt_provider.transcribe(audio_buffer)
            self.context.preliminary_transcript = transcript
            
            logger.info(f"🔍 Query validation transcript: '{transcript}'")
            
            # Check for bypass keywords
            transcript_lower = transcript.lower()
            for keyword in self.behavior.allow_keywords:
                if keyword in transcript_lower:
                    logger.info(f"✅ Query validation PASSED: Keyword '{keyword}' detected")
                    return True
            
            # Check word count
            words = transcript.split()
            if len(words) < self.behavior.min_word_count:
                logger.info(
                    f"❌ Query validation FAILED: Too few words ({len(words)} < {self.behavior.min_word_count})"
                )
                self.context.rejection_reason = f"Insufficient words: {len(words)} < {self.behavior.min_word_count}"
                return False
            
            # Check for meaningless utterances (um, uh, etc.)
            meaningless = {"um", "uh", "er", "ah", "hmm", "mhm"}
            if len(words) <= 2 and all(w.lower() in meaningless for w in words):
                logger.info(f"❌ Query validation FAILED: Only meaningless filler words")
                self.context.rejection_reason = "Only filler words detected"
                return False
            
            # All checks passed
            logger.info(f"✅ Query validation PASSED: {len(words)} words, meaningful content")
            return True
            
        except Exception as e:
            logger.error(f"❌ Query validation error: {e}", exc_info=True)
            self.context.rejection_reason = f"STT error: {str(e)}"
            return False
