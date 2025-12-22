"""
Persona-specific behavioral parameters for intelligent barge-in and turn-taking.

This module defines behavior strategies for different persona types, controlling:
- Barge-in sensitivity (how easily user can interrupt)
- Validation strictness (speech validation thresholds)
- Query confidence requirements
- Grace periods and timeout values

Each persona type (interviewer, technical-expert, helpful-assistant) has distinct
behavioral characteristics that influence interruption handling.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BargeinBehavior:
    """Defines barge-in behavioral parameters for a persona type.
    
    Attributes:
        persona_type: Persona identifier (e.g., "technical-interviewer")
        
        # Stage 1: Speech Validation Parameters
        min_energy_threshold: Minimum peak energy above noise floor (amplitude units)
        min_snr_db: Minimum Signal-to-Noise Ratio in decibels
        min_consecutive_frames: Minimum consecutive frames of voiced speech (60ms/frame typical)
        speech_validation_duration_ms: Minimum duration of sustained speech to pass validation
        
        # Stage 2: Query Validation Parameters
        buffer_duration_ms: Duration of audio to buffer for preliminary query check
        confidence_threshold: Minimum confidence score (0.0-1.0) for query validation
        min_word_count: Minimum words required in preliminary transcript
        allow_keywords: List of keywords that bypass full validation (e.g., ["stop", "wait"])
        
        # Turn-Taking Parameters
        post_ai_grace_ms: Grace period after AI stops speaking before accepting input
        max_ai_response_time_s: Maximum allowed AI speech duration before forcing accept
        allow_mid_sentence_barge: Whether to allow interruption mid-sentence
        
        # Behavioral Flags
        enable_proactive_follow_up: AI can ask follow-up during grace period
        enable_clarification_barge: User can interrupt during clarification requests
    """
    persona_type: str
    
    # Stage 1: Speech Validation
    min_energy_threshold: int = 1800
    min_snr_db: float = 12.0
    min_consecutive_frames: int = 10  # 200ms @ 50fps
    speech_validation_duration_ms: int = 300
    
    # Stage 2: Query Validation
    buffer_duration_ms: int = 1500
    confidence_threshold: float = 0.7
    min_word_count: int = 2
    allow_keywords: list = None
    
    # Turn-Taking
    post_ai_grace_ms: int = 1200
    max_ai_response_time_s: int = 45
    allow_mid_sentence_barge: bool = True
    
    # Behavioral Flags
    enable_proactive_follow_up: bool = False
    enable_clarification_barge: bool = True
    
    def __post_init__(self):
        if self.allow_keywords is None:
            self.allow_keywords = ["stop", "wait", "hold on"]


# ==================================================================================
# PERSONA BEHAVIOR PROFILES
# ==================================================================================

INTERVIEWER_BEHAVIOR = BargeinBehavior(
    persona_type="technical-interviewer",
    
    # AGGRESSIVE BARGE-IN: Interviewer expects short, frequent exchanges
    min_energy_threshold=1200,  # Lower threshold (easier to interrupt)
    min_snr_db=10.0,  # More permissive SNR
    min_consecutive_frames=8,  # Shorter validation (160ms)
    speech_validation_duration_ms=250,  # Fast validation
    
    # LENIENT QUERY VALIDATION: Accept brief responses
    buffer_duration_ms=1000,  # Shorter buffer
    confidence_threshold=0.6,  # Lower confidence OK
    min_word_count=1,  # Single words accepted (e.g., "yes", "no")
    allow_keywords=["stop", "wait", "no", "yes", "correct", "wrong"],
    
    # RESPONSIVE TURN-TAKING: Quick back-and-forth
    post_ai_grace_ms=800,  # Shorter grace period
    max_ai_response_time_s=30,  # Shorter responses expected
    allow_mid_sentence_barge=True,
    
    # PROACTIVE BEHAVIOR: Interviewer drives conversation
    enable_proactive_follow_up=True,
    enable_clarification_barge=True
)


TECHNICAL_EXPERT_BEHAVIOR = BargeinBehavior(
    persona_type="technical-expert",
    
    # CONSERVATIVE BARGE-IN: Expert gives detailed explanations
    min_energy_threshold=2000,  # Higher threshold (harder to interrupt)
    min_snr_db=13.0,  # Stricter SNR requirement
    min_consecutive_frames=12,  # Longer validation (240ms)
    speech_validation_duration_ms=400,  # More thorough validation
    
    # STRICT QUERY VALIDATION: Ensure user has a real question
    buffer_duration_ms=1800,  # Longer buffer for complete thoughts
    confidence_threshold=0.75,  # Higher confidence required
    min_word_count=3,  # Multi-word phrases required
    allow_keywords=["stop", "wait", "question"],
    
    # PATIENT TURN-TAKING: Allow expert to complete thoughts
    post_ai_grace_ms=1500,  # Longer grace period
    max_ai_response_time_s=60,  # Allow longer explanations
    allow_mid_sentence_barge=False,  # Protect mid-sentence flow
    
    # PASSIVE BEHAVIOR: User drives clarifications
    enable_proactive_follow_up=False,
    enable_clarification_barge=False  # Don't interrupt during setup
)


HELPFUL_ASSISTANT_BEHAVIOR = BargeinBehavior(
    persona_type="helpful-assistant",
    
    # BALANCED BARGE-IN: Standard conversational flow
    min_energy_threshold=1600,  # Moderate threshold
    min_snr_db=11.0,  # Balanced SNR
    min_consecutive_frames=10,  # Standard validation (200ms)
    speech_validation_duration_ms=300,  # Standard duration
    
    # MODERATE QUERY VALIDATION: Accept clear questions
    buffer_duration_ms=1500,  # Standard buffer
    confidence_threshold=0.7,  # Moderate confidence
    min_word_count=2,  # Typical phrases
    allow_keywords=["stop", "wait", "hold on", "question"],
    
    # STANDARD TURN-TAKING: Natural conversation flow
    post_ai_grace_ms=1200,  # Standard grace
    max_ai_response_time_s=45,  # Standard response time
    allow_mid_sentence_barge=True,
    
    # BALANCED BEHAVIOR: Reactive but responsive
    enable_proactive_follow_up=False,
    enable_clarification_barge=True
)


EMPATHETIC_COACH_BEHAVIOR = BargeinBehavior(
    persona_type="empathetic-coach",
    
    # VERY CONSERVATIVE BARGE-IN: Allow user to fully express
    min_energy_threshold=2200,  # High threshold (coaching requires patience)
    min_snr_db=14.0,  # Strict SNR
    min_consecutive_frames=15,  # Long validation (300ms)
    speech_validation_duration_ms=500,  # Very thorough
    
    # VERY STRICT QUERY VALIDATION: Ensure intentional interruption
    buffer_duration_ms=2000,  # Long buffer for emotional expression
    confidence_threshold=0.8,  # High confidence
    min_word_count=4,  # Complete phrases required
    allow_keywords=["stop", "wait", "pause"],
    
    # PATIENT TURN-TAKING: Give space for reflection
    post_ai_grace_ms=2000,  # Very long grace (allow thinking time)
    max_ai_response_time_s=50,  # Allow thoughtful responses
    allow_mid_sentence_barge=False,  # Protect reflective moments
    
    # GENTLE BEHAVIOR: Non-intrusive coaching
    enable_proactive_follow_up=False,
    enable_clarification_barge=False
)


# ==================================================================================
# BEHAVIOR REGISTRY
# ==================================================================================

BEHAVIOR_REGISTRY: Dict[str, BargeinBehavior] = {
    "technical-interviewer": INTERVIEWER_BEHAVIOR,
    "technical-expert": TECHNICAL_EXPERT_BEHAVIOR,
    "helpful-assistant": HELPFUL_ASSISTANT_BEHAVIOR,
    "empathetic-coach": EMPATHETIC_COACH_BEHAVIOR,
    "default": HELPFUL_ASSISTANT_BEHAVIOR,  # Safe default
}


def get_behavior_for_persona(persona_slug: str, persona_config: Dict[str, Any]) -> BargeinBehavior:
    """Resolve barge-in behavior for a given persona.
    
    Args:
        persona_slug: Persona identifier (e.g., "technical-interviewer-xj3k2")
        persona_config: Full persona configuration dict
        
    Returns:
        BargeinBehavior instance with persona-specific parameters
        
    Strategy:
        1. Check persona_config for 'source_template_id' (e.g., "technical-interviewer")
        2. Fall back to matching slug prefix
        3. Use 'default' if no match found
    """
    # Try metadata source_template_id first (most reliable)
    metadata = persona_config.get('metadata', {})
    template_id = metadata.get('source_template_id')
    
    if template_id and template_id in BEHAVIOR_REGISTRY:
        behavior = BEHAVIOR_REGISTRY[template_id]
        logger.info(f"✅ Loaded barge-in behavior for persona '{persona_slug}' from template '{template_id}'")
        return behavior
    
    # Try matching slug prefix (e.g., "technical-interviewer-abc123" -> "technical-interviewer")
    for key in BEHAVIOR_REGISTRY.keys():
        if persona_slug.startswith(key):
            behavior = BEHAVIOR_REGISTRY[key]
            logger.info(f"✅ Loaded barge-in behavior for persona '{persona_slug}' via slug prefix match '{key}'")
            return behavior
    
    # Fallback to default
    logger.warning(f"⚠️ No specific barge-in behavior found for persona '{persona_slug}', using default")
    return BEHAVIOR_REGISTRY["default"]


def log_behavior_config(behavior: BargeinBehavior):
    """Log the active barge-in behavior configuration for debugging."""
    logger.info("=" * 70)
    logger.info(f"🎭 BARGE-IN BEHAVIOR CONFIG: {behavior.persona_type}")
    logger.info("=" * 70)
    logger.info(f"  Stage 1 (Speech Validation):")
    logger.info(f"    - Min energy: {behavior.min_energy_threshold}")
    logger.info(f"    - Min SNR: {behavior.min_snr_db} dB")
    logger.info(f"    - Min frames: {behavior.min_consecutive_frames} ({behavior.min_consecutive_frames * 20}ms)")
    logger.info(f"    - Validation duration: {behavior.speech_validation_duration_ms}ms")
    logger.info(f"  Stage 2 (Query Validation):")
    logger.info(f"    - Buffer duration: {behavior.buffer_duration_ms}ms")
    logger.info(f"    - Confidence threshold: {behavior.confidence_threshold}")
    logger.info(f"    - Min words: {behavior.min_word_count}")
    logger.info(f"    - Allow keywords: {behavior.allow_keywords}")
    logger.info(f"  Turn-Taking:")
    logger.info(f"    - Post-AI grace: {behavior.post_ai_grace_ms}ms")
    logger.info(f"    - Max AI speech: {behavior.max_ai_response_time_s}s")
    logger.info(f"    - Mid-sentence barge: {behavior.allow_mid_sentence_barge}")
    logger.info("=" * 70)
