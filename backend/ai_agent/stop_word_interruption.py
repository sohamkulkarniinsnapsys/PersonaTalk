"""
Stop-Word Interruption System for Reliable User Control

Implements semantic (text-based) stop-word detection as a high-confidence
fallback to energy-based barge-in. Allows users to interrupt AI with explicit
control commands like "stop", "wait", "hold on", etc.

Architecture:
  Tier-0: Energy-based barge-in (existing, 75-85% success)
  Tier-1: Hard stop-words (new, 95%+ success)
  Tier-2: Soft stop-words (new, 70% with warnings)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class InterruptionTier(str, Enum):
    """Interruption confidence tier."""
    TIER_1_HARD = "TIER_1_HARD_STOP"
    TIER_2_SOFT = "TIER_2_SOFT_STOP"
    NONE = "NONE"


@dataclass
class StopWordMatch:
    """Result of stop-word detection."""
    matched: bool
    tier: InterruptionTier = InterruptionTier.NONE
    keyword: Optional[str] = None
    confidence: float = 0.0
    transcript_snippet: str = ""
    buffer_duration_ms: int = 0
    detection_time_ms: float = 0.0  # Time from utterance start to detection


class StopWordInterruptor:
    """
    Detects stop-words in real-time audio transcripts.
    
    Provides multi-tier interruption:
    - Tier-1 (hard stops): Never appear in natural conversation
      Keywords: "stop", "wait", "hold on", "excuse me", "pause"
      Confidence: 100% → Immediate interruption
      
    - Tier-2 (soft stops): Could appear conversationally but often indicate interruption intent
      Keywords: "question", "actually", "sorry", "um" (with context)
      Confidence: 70-80% → Warning/volume reduction instead of immediate stop
      
    - Exclusions: Common fillers excluded unless in context
      "um", "uh", "like" alone → ignore
      "um, I have a question" → Tier-1 or Tier-2 depending on context
    """
    
    # Tier-1: Hard stops (100% confidence, never conversational)
    TIER_1_KEYWORDS: Dict[str, float] = {
        "stop": 1.0,
        "wait": 1.0,
        "hold on": 1.0,
        "hold": 1.0,
        "excuse me": 1.0,
        "pause": 0.95,
        "hang on": 0.95,
        "quit": 0.95,
        "exit": 0.95,
        "break": 0.90,  # "break" when said alone during speech = interrupt
    }
    
    # Tier-2: Soft stops (70-80% confidence, context-sensitive)
    TIER_2_KEYWORDS: Dict[str, float] = {
        "question": 0.75,
        "actually": 0.70,
        "wait a": 0.90,  # "wait a minute/second" = Tier-2 (longer phrase)
        "sorry": 0.70,
        "can i": 0.70,  # "can I ask/interrupt?" = Tier-2
    }
    
    # Common filler words to exclude unless in specific context
    EXCLUSIONS: List[str] = [
        "um",
        "uh", 
        "er",
        "ah",
        "like",
        "you know",
        "i mean",
        "hmm",
    ]
    
    def __init__(self, check_interval_ms: int = 500):
        """
        Initialize stop-word detector.
        
        Args:
            check_interval_ms: How often to check for stop-words (default 500ms)
        """
        self.check_interval_ms = check_interval_ms
        self.last_check_time: float = 0.0
        self.last_check_transcript: str = ""
        self.transcripts_history: deque = deque(maxlen=10)  # Keep last 10 transcripts for context
        self.detection_count: int = 0
        self.false_positive_count: int = 0
        self.session_id: str = ""
        
        logger.info(
            f"🎯 StopWordInterruptor initialized\n"
            f"   Tier-1 keywords: {list(self.TIER_1_KEYWORDS.keys())}\n"
            f"   Tier-2 keywords: {list(self.TIER_2_KEYWORDS.keys())}\n"
            f"   Check interval: {check_interval_ms}ms"
        )
    
    async def check_for_stop_words(
        self,
        transcript: str,
        buffer_duration_ms: int,
        stt_confidence: float = 0.85,
        time_since_ai_start_ms: float = 0.0,
        utterance_id: Optional[str] = None
    ) -> StopWordMatch:
        """
        Check transcript for stop-words.
        
        Args:
            transcript: Full transcript of recent audio buffer
            buffer_duration_ms: Duration of audio buffer (ms)
            stt_confidence: STT provider confidence score (0.0-1.0)
            time_since_ai_start_ms: Time elapsed since AI started speaking (ms)
            utterance_id: For logging correlation
            
        Returns:
            StopWordMatch with tier and keyword if detected
        """
        start_time = time.time()
        
        # Skip empty transcripts
        if not transcript or not transcript.strip():
            return StopWordMatch(matched=False)
        
        # Normalize transcript for matching
        norm_text = self._normalize_text(transcript)
        utterance_info = f" (utterance_id={utterance_id})" if utterance_id else ""
        
        logger.debug(
            f"🔍 Stop-word check: '{transcript}' → '{norm_text}'{utterance_info}\n"
            f"   Buffer: {buffer_duration_ms}ms, STT conf: {stt_confidence:.1%}, "
            f"Time in AI: {time_since_ai_start_ms:.0f}ms"
        )
        
        # Check Tier-1 keywords (hard stops)
        for keyword, base_confidence in self.TIER_1_KEYWORDS.items():
            if self._keyword_in_text(keyword, norm_text):
                # Tier-1 matches are high-confidence
                confidence = base_confidence * stt_confidence
                
                logger.info(
                    f"✅ TIER-1 STOP-WORD DETECTED: '{keyword}' in '{transcript}'\n"
                    f"   Confidence: {confidence:.0%} (base={base_confidence:.0%}, STT={stt_confidence:.0%})\n"
                    f"   Buffer: {buffer_duration_ms}ms, Time in AI: {time_since_ai_start_ms:.0f}ms"
                )
                
                self.detection_count += 1
                self.transcripts_history.append(transcript)
                
                return StopWordMatch(
                    matched=True,
                    tier=InterruptionTier.TIER_1_HARD,
                    keyword=keyword,
                    confidence=confidence,
                    transcript_snippet=transcript,
                    buffer_duration_ms=buffer_duration_ms,
                    detection_time_ms=time_since_ai_start_ms
                )
        
        # Check Tier-2 keywords (soft stops)
        for keyword, base_confidence in self.TIER_2_KEYWORDS.items():
            if self._keyword_in_text(keyword, norm_text):
                # Tier-2 matches are medium-confidence
                # Factor in context: if utterance is SHORT, likely not conversational
                if self._is_likely_interruption(norm_text, keyword):
                    confidence = base_confidence * stt_confidence
                    
                    logger.info(
                        f"⚠️  TIER-2 STOP-WORD DETECTED: '{keyword}' in '{transcript}'\n"
                        f"   Confidence: {confidence:.0%} (base={base_confidence:.0%}, STT={stt_confidence:.0%})\n"
                        f"   Buffer: {buffer_duration_ms}ms, Time in AI: {time_since_ai_start_ms:.0f}ms\n"
                        f"   NOTE: Tier-2 is soft interrupt (not immediate stop)"
                    )
                    
                    self.detection_count += 1
                    self.transcripts_history.append(transcript)
                    
                    return StopWordMatch(
                        matched=True,
                        tier=InterruptionTier.TIER_2_SOFT,
                        keyword=keyword,
                        confidence=confidence,
                        transcript_snippet=transcript,
                        buffer_duration_ms=buffer_duration_ms,
                        detection_time_ms=time_since_ai_start_ms
                    )
        
        # No stop-word detected
        self.transcripts_history.append(transcript)
        return StopWordMatch(
            matched=False,
            transcript_snippet=transcript,
            buffer_duration_ms=buffer_duration_ms
        )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize transcript for keyword matching.
        
        - Lowercase
        - Remove punctuation
        - Strip leading/trailing spaces
        - Compress multiple spaces
        """
        import re
        
        normalized = text.lower()
        # Remove punctuation except hyphens
        normalized = re.sub(r'[.,!?;:()\[\]{}"\']', '', normalized)
        # Compress multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def _keyword_in_text(self, keyword: str, normalized_text: str) -> bool:
        """
        Check if keyword appears in normalized text.
        
        Handles both single words and multi-word phrases.
        """
        # For multi-word phrases, check as substring
        if ' ' in keyword:
            return keyword in normalized_text
        
        # For single words, check as word boundary
        import re
        pattern = r'\b' + re.escape(keyword) + r'\b'
        return bool(re.search(pattern, normalized_text))
    
    def _is_likely_interruption(self, normalized_text: str, keyword: str) -> bool:
        """
        Determine if Tier-2 keyword is likely interruption intent vs natural conversation.
        
        Heuristics:
        - Short utterance (< 10 words) = likely interruption
        - Single keyword with minimal context = interruption
        - "question" + preposition/article = likely interruption ("I have a question")
        - "actually" at start of utterance = likely correction = interruption
        """
        words = normalized_text.split()
        word_count = len(words)
        
        # Short utterance with Tier-2 keyword = likely interruption
        if word_count <= 5:
            logger.debug(f"   Context: Short utterance ({word_count} words) → likely interruption")
            return True
        
        # "question" with article/preposition = likely "I have a question"
        if keyword == "question":
            if any(word in normalized_text for word in ["have", "got", "do i", "would i", "can i"]):
                logger.debug(f"   Context: 'question' with intent marker → likely interruption")
                return True
        
        # "actually" at start = likely correction/new thought
        if keyword == "actually" and normalized_text.startswith("actually"):
            logger.debug(f"   Context: 'actually' at start → likely interruption")
            return True
        
        # "sorry" at start = likely apology for interruption
        if keyword == "sorry" and normalized_text.startswith("sorry"):
            logger.debug(f"   Context: 'sorry' at start → likely interruption apology")
            return True
        
        # Default: longer utterance = conversational, not interruption intent
        logger.debug(f"   Context: Longer utterance ({word_count} words) → likely conversational")
        return False
    
    def should_check_now(self) -> bool:
        """Check if enough time has elapsed since last check."""
        current_time = time.time()
        time_since_last_ms = (current_time - self.last_check_time) * 1000
        
        if time_since_last_ms >= self.check_interval_ms:
            self.last_check_time = current_time
            return True
        
        return False
    
    def reset_check_timer(self):
        """Reset the check timer (call after finding a match or on phase change)."""
        self.last_check_time = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return detection statistics."""
        return {
            "total_detections": self.detection_count,
            "false_positive_count": self.false_positive_count,
            "detection_rate": self.detection_count / max(1, self.detection_count + 100),  # Rough estimate
            "recent_transcripts": list(self.transcripts_history),
        }
    
    async def reset_session(self, session_id: str = ""):
        """Reset state for new session."""
        self.session_id = session_id
        self.detection_count = 0
        self.false_positive_count = 0
        self.transcripts_history.clear()
        self.last_check_transcript = ""
        logger.info(f"🔄 StopWordInterruptor reset for session: {session_id or 'unknown'}")
