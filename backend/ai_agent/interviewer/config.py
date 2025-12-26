"""
Dynamic configuration schema for interviewer persona.
All interview behavior is driven by this config - no hard-coding.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ScoringRules:
    """Deterministic scoring rules for each outcome combination."""
    correct_first_attempt: int = 9
    partial_then_correct: int = 7
    partial_then_failed: int = 5
    incorrect_then_correct: int = 4
    incorrect_then_failed: int = 1
    
    # Thresholds for classification
    concept_coverage_for_correct: float = 0.6  # 60% of key concepts
    concept_coverage_for_partial: float = 0.3  # 30% for partial


@dataclass
class HintingRules:
    """Rules for when and how to give hints."""
    max_hints_per_question: int = 1
    hint_must_be_conceptual: bool = True  # Never reveal keywords/formulas
    hint_max_length_words: int = 25
    forbidden_in_hints: List[str] = field(default_factory=lambda: [
        "the answer is", "correct answer", "solution is", "formula is"
    ])


@dataclass
class RetryRules:
    """Rules for retry behavior."""
    max_attempts_per_question: int = 2
    idk_is_incorrect: bool = True  # "I don't know" treated as wrong answer
    idk_keywords: List[str] = field(default_factory=lambda: [
        "i don't know", "dont know", "don't know", "no idea", "not sure", "idk"
    ])


@dataclass
class FeedbackRules:
    """Final evaluation feedback generation rules."""
    excellent_threshold: int = 75  # >= 75% is excellent
    good_threshold: int = 50       # 50-74% is good
    # < 50% is weak
    max_weak_areas_to_mention: int = 3


@dataclass
class QuestionDefinition:
    """Single question with canonical answer and metadata."""
    text: str
    answer: str  # Canonical correct answer
    concepts: List[str]  # Key concepts for evaluation
    hint: str  # High-level conceptual hint only
    tech: str = "general"
    difficulty: str = "basic"


@dataclass
class InterviewConfig:
    """Complete runtime configuration for interviewer persona.
    
    This replaces all hard-coded behavior. Changes to interview flow
    require only config updates, not code changes.
    """
    # Question bank organized by tech/difficulty
    questions: Dict[str, Dict[str, List[QuestionDefinition]]] = field(default_factory=dict)
    
    # Behavioral rules
    scoring: ScoringRules = field(default_factory=ScoringRules)
    hinting: HintingRules = field(default_factory=HintingRules)
    retry: RetryRules = field(default_factory=RetryRules)
    feedback: FeedbackRules = field(default_factory=FeedbackRules)
    
    # Interview structure
    total_questions: int = 6
    default_tech: str = "general"
    
    # Lifecycle enforcement
    require_terminal_state: bool = True  # Never skip unresolved questions
    strict_hint_limit: bool = True       # Only one hint per question
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InterviewConfig:
        """Load from JSON-serializable dict."""
        # Parse questions
        questions_raw = data.get("questions", {})
        questions = {}
        for tech, levels in questions_raw.items():
            questions[tech] = {}
            for level, qlist in levels.items():
                questions[tech][level] = [
                    QuestionDefinition(
                        text=q["text"],
                        answer=q["answer"],
                        concepts=q.get("concepts", []),
                        hint=q.get("hint", ""),
                        tech=tech,
                        difficulty=level
                    )
                    for q in qlist
                ]
        
        scoring_data = data.get("scoring", {})
        hinting_data = data.get("hinting", {})
        retry_data = data.get("retry", {})
        feedback_data = data.get("feedback", {})
        
        return cls(
            questions=questions,
            scoring=ScoringRules(**scoring_data) if scoring_data else ScoringRules(),
            hinting=HintingRules(**hinting_data) if hinting_data else HintingRules(),
            retry=RetryRules(**retry_data) if retry_data else RetryRules(),
            feedback=FeedbackRules(**feedback_data) if feedback_data else FeedbackRules(),
            total_questions=data.get("total_questions", 6),
            default_tech=data.get("default_tech", "general"),
            require_terminal_state=data.get("require_terminal_state", True),
            strict_hint_limit=data.get("strict_hint_limit", True),
        )


# Default configuration - can be overridden per persona instance
DEFAULT_INTERVIEWER_CONFIG = InterviewConfig(
    questions={
        "general": {
            "basic": [
                QuestionDefinition(
                    text="In simple terms, what is a REST API and how is it different from RPC?",
                    answer=(
                        "A REST API exposes resources over HTTP using standard verbs like GET, POST, PUT, DELETE. "
                        "Clients operate on resource representations via stateless requests, where URLs identify resources. "
                        "RPC focuses on calling functions or procedures, modeling operations as method calls rather than resources."
                    ),
                    concepts=["http verbs", "resources", "stateless", "urls", "rpc is function calls"],
                    hint="Think about resources and standard HTTP verbs versus calling functions."
                ),
                QuestionDefinition(
                    text="What do HTTP 200 and 404 status codes mean?",
                    answer="200 means a request succeeded. 404 means the requested resource was not found.",
                    concepts=["200 ok", "success", "404 not found", "resource missing"],
                    hint="One signals success; the other indicates the resource isn't there."
                ),
                QuestionDefinition(
                    text="What is the time complexity of binary search and why?",
                    answer="O(log n) because each step halves the remaining search interval in a sorted array.",
                    concepts=["log n", "halve", "sorted"],
                    hint="Consider how many times you can halve the search space."
                ),
            ],
            "moderate": [
                QuestionDefinition(
                    text="How would you design a rate limiter for an API? Mention one algorithm.",
                    answer=(
                        "Use token bucket or leaky bucket with a shared store like Redis to track tokens per identity. "
                        "Requests consume tokens; tokens refill over time to enforce a steady rate."
                    ),
                    concepts=["token bucket", "leaky bucket", "shared store", "refill", "identity"],
                    hint="Think about tokens, a shared counter, and refill over time."
                ),
            ],
            "advanced": [
                QuestionDefinition(
                    text="Explain the CAP theorem trade-offs for distributed systems.",
                    answer=(
                        "In the presence of a network partition, you must choose between Consistency and Availability. "
                        "Systems can at most provide any two of Consistency, Availability, and Partition tolerance."
                    ),
                    concepts=["consistency", "availability", "partition tolerance", "trade-off"],
                    hint="During a partition you make a choice; which two can you keep?"
                ),
            ],
        },
        "python": {
            "basic": [
                QuestionDefinition(
                    text="What are lists vs tuples in Python, and when use each?",
                    answer=(
                        "Lists are mutable sequences suitable for items that change. "
                        "Tuples are immutable, often used for fixed collections or as dict keys."
                    ),
                    concepts=["mutable", "immutable", "sequence", "use cases"],
                    hint="One changes, one doesn't; think about when you'd want that."
                ),
            ]
        },
    },
    total_questions=6,
)
