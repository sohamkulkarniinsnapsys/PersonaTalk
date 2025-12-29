from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ScoringRules

# STRICTLY persona-scoped evaluator. Uses simple semantic coverage + optional LLM adjudication

class Classification:
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def keyword_coverage(
    user_answer: str,
    concepts: List[str],
    correct_threshold: float = 0.6
) -> Tuple[int, int]:
    """Count concept hits in the user's answer.
    This is deterministic and fast; good first-pass filter.
    
    Args:
        user_answer: Candidate's response
        concepts: Key concepts to check
        correct_threshold: Fraction of concepts needed for "correct" (default 0.6 = 60%)
    """
    s = _normalize(user_answer)
    hits = 0
    for c in concepts or []:
        if c and c.lower() in s:
            hits += 1
    needed = max(1, math.ceil(len(concepts or []) * correct_threshold))
    return hits, needed


def classify(
    user_answer: str,
    reference_answer: str,
    concepts: List[str],
    correct_threshold: float = 0.6,
    partial_threshold: float = 0.3
) -> Dict:
    """Classify into correct/partial/incorrect with config-driven rules.
    
    ENHANCED: More lenient evaluation - considers semantic similarity and directional correctness.
    
    Args:
        user_answer: Candidate's response
        reference_answer: Canonical correct answer
        concepts: Key concepts to check
        correct_threshold: Fraction needed for "correct" (default 0.6)
        partial_threshold: Fraction needed for "partial" (default 0.3)
    """
    s = _normalize(user_answer)
    if not s:
        return {"label": Classification.INCORRECT, "reason": "empty"}

    hits, _ = keyword_coverage(s, concepts, correct_threshold)
    total_concepts = len(concepts or [1])
    coverage = hits / max(total_concepts, 1)
    
    # ENHANCED: Be more lenient - 50% coverage is acceptable for "correct" in many cases
    # Original threshold was too strict (60%)
    lenient_correct_threshold = max(0.5, correct_threshold - 0.1)  # Lower by 10%
    
    if coverage >= lenient_correct_threshold:
        label = Classification.CORRECT
    elif coverage >= partial_threshold:
        label = Classification.PARTIAL
    else:
        # ENHANCED: Better semantic overlap detection
        ref = _normalize(reference_answer)
        # Check for meaningful phrase overlap (not just single words)
        ref_tokens = ref.split()
        # Look for 2-3 word phrases, not just single words
        overlap_score = 0
        for i in range(len(ref_tokens) - 1):
            bigram = " ".join(ref_tokens[i:i+2])
            if bigram in s:
                overlap_score += 2
        # Also check single important words
        for tok in ref_tokens[:5]:
            if len(tok) > 3 and tok in s:  # Only meaningful words (>3 chars)
                overlap_score += 1
        
        label = Classification.PARTIAL if overlap_score >= 2 else Classification.INCORRECT
    
    return {"label": label, "hits": hits, "coverage": coverage, "needed": int(total_concepts * lenient_correct_threshold)}


def score_for(
    first_label: str,
    second_label: str | None,
    scoring_rules: Optional["ScoringRules"] = None
) -> Tuple[int, str]:
    """Compute score and provide derivation text following config-driven rules.
    
    Args:
        first_label: Classification of first attempt
        second_label: Classification of second attempt (or None if only one attempt)
        scoring_rules: Config-driven scoring rules (uses defaults if not provided)
        
    Returns:
        Tuple of (score, derivation_text)
    """
    # Use defaults if no rules provided
    if scoring_rules is None:
        from .config import ScoringRules
        scoring_rules = ScoringRules()
    
    if first_label == Classification.CORRECT:
        return scoring_rules.correct_first_attempt, "correct on first attempt"
    if first_label == Classification.PARTIAL and second_label == Classification.CORRECT:
        return scoring_rules.partial_then_correct, "partial then correct after hint"
    if first_label == Classification.PARTIAL and (second_label in (Classification.PARTIAL, Classification.INCORRECT, None)):
        return scoring_rules.partial_then_failed, "partial then not correct after hint"
    if first_label == Classification.INCORRECT and second_label == Classification.CORRECT:
        return scoring_rules.incorrect_then_correct, "incorrect then correct after hint"
    return scoring_rules.incorrect_then_failed, "incorrect both attempts"
