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
    
    if coverage >= correct_threshold:
        label = Classification.CORRECT
    elif coverage >= partial_threshold:
        label = Classification.PARTIAL
    else:
        # allow semantic overlap via inclusion of key phrases from reference
        ref = _normalize(reference_answer)
        overlap = 1 if any(tok in s for tok in ref.split()[:5]) else 0
        label = Classification.PARTIAL if overlap else Classification.INCORRECT
    
    return {"label": label, "hits": hits, "coverage": coverage, "needed": int(total_concepts * correct_threshold)}


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
