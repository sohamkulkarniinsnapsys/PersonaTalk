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
    concept_coverage_for_correct: float = 0.5  # 50% of key concepts (lenient)
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
    difficulty: str = "beginner"


@dataclass
class InterviewConfig:
    """Complete runtime configuration for interviewer persona.
    
    This replaces all hard-coded behavior. Changes to interview flow
    require only config updates, not code changes.
    
    DYNAMIC GENERATION SUPPORT:
    - If use_dynamic_generation=True, questions are generated at runtime by LLM
    - If use_dynamic_generation=False, falls back to hard-coded 'questions' bank
    """
    # Question bank organized by tech/difficulty (fallback for static mode)
    # Keys: tech (domain) -> difficulty level -> list of questions
    questions: Dict[str, Dict[str, List[QuestionDefinition]]] = field(default_factory=dict)
    
    # Behavioral rules
    scoring: ScoringRules = field(default_factory=ScoringRules)
    hinting: HintingRules = field(default_factory=HintingRules)
    retry: RetryRules = field(default_factory=RetryRules)
    feedback: FeedbackRules = field(default_factory=FeedbackRules)
    
    # Interview structure
    total_questions: int = 10  # 4 beginner + 3 intermediate + 3 advanced
    default_tech: str = "general"
    
    # Dynamic generation settings
    use_dynamic_generation: bool = True  # Enable runtime question generation
    difficulty_distribution: Dict[str, int] = field(default_factory=lambda: {
        "beginner": 4,
        "intermediate": 3,
        "advanced": 3
    })
    
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
        
        difficulty_dist = data.get("difficulty_distribution", {
            "beginner": 4,
            "intermediate": 3,
            "advanced": 3
        })
        
        return cls(
            questions=questions,
            scoring=ScoringRules(**scoring_data) if scoring_data else ScoringRules(),
            hinting=HintingRules(**hinting_data) if hinting_data else HintingRules(),
            retry=RetryRules(**retry_data) if retry_data else RetryRules(),
            feedback=FeedbackRules(**feedback_data) if feedback_data else FeedbackRules(),
            total_questions=data.get("total_questions", 10),
            default_tech=data.get("default_tech", "general"),
            use_dynamic_generation=data.get("use_dynamic_generation", True),
            difficulty_distribution=difficulty_dist,
            require_terminal_state=data.get("require_terminal_state", True),
            strict_hint_limit=data.get("strict_hint_limit", True),
        )


# Default configuration - uses dynamic generation by default
DEFAULT_INTERVIEWER_CONFIG = InterviewConfig(
    # Minimal fallback questions (only if dynamic generation fails)
    questions={
        "general": {
            "beginner": [
                QuestionDefinition(
                    text="In simple terms, what is a REST API and how is it different from RPC?",
                    answer=(
                        "A REST API exposes resources over HTTP using standard verbs like GET, POST, PUT, DELETE. "
                        "Clients operate on resource representations via stateless requests, where URLs identify resources. "
                        "RPC focuses on calling functions or procedures, modeling operations as method calls rather than resources."
                    ),
                    concepts=["http verbs", "resources", "stateless", "urls", "rpc"],
                    hint="Think about resources and standard HTTP verbs versus calling functions.",
                    tech="general",
                    difficulty="beginner"
                ),
                QuestionDefinition(
                    text="What do HTTP 200 and 404 status codes mean?",
                    answer="200 means a request succeeded. 404 means the requested resource was not found.",
                    concepts=["200 ok", "success", "404 not found", "resource missing"],
                    hint="One signals success; the other indicates the resource isn't there.",
                    tech="general",
                    difficulty="beginner"
                ),
                QuestionDefinition(
                    text="What is the time complexity of binary search and why?",
                    answer="O(log n) because each step halves the remaining search interval in a sorted array.",
                    concepts=["log n", "halve", "sorted", "divide"],
                    hint="Consider how many times you can halve the search space.",
                    tech="general",
                    difficulty="beginner"
                ),
                QuestionDefinition(
                    text="What is the difference between a process and a thread?",
                    answer="A process is an independent program with its own memory space. A thread is a lightweight execution unit within a process that shares memory with other threads.",
                    concepts=["process", "thread", "memory space", "independence", "shared memory"],
                    hint="Think about memory isolation versus sharing.",
                    tech="general",
                    difficulty="beginner"
                ),
            ],
            "intermediate": [
                QuestionDefinition(
                    text="How would you design a rate limiter for an API? Mention one algorithm.",
                    answer=(
                        "Use token bucket or leaky bucket with a shared store like Redis to track tokens per identity. "
                        "Requests consume tokens; tokens refill over time to enforce a steady rate."
                    ),
                    concepts=["token bucket", "leaky bucket", "shared store", "refill", "algorithm"],
                    hint="Think about tokens, a shared counter, and refill over time.",
                    tech="general",
                    difficulty="intermediate"
                ),
                QuestionDefinition(
                    text="Explain how a hash table works and its average time complexity for lookups.",
                    answer="A hash table uses a hash function to map keys to array indices. On average, lookups are O(1). Collisions are handled via chaining or open addressing.",
                    concepts=["hash function", "array indices", "O(1)", "collisions", "lookup"],
                    hint="Think about how keys get converted to positions.",
                    tech="general",
                    difficulty="intermediate"
                ),
                QuestionDefinition(
                    text="What is the difference between SQL and NoSQL databases?",
                    answer="SQL uses structured schemas with tables and ACID transactions. NoSQL is schema-less, horizontally scalable, and optimized for specific data models like document, key-value, or graph.",
                    concepts=["schema", "ACID", "scaling", "data models", "relational"],
                    hint="Think about structure, transactions, and scaling approaches.",
                    tech="general",
                    difficulty="intermediate"
                ),
            ],
            "advanced": [
                QuestionDefinition(
                    text="Explain the CAP theorem trade-offs for distributed systems.",
                    answer=(
                        "In the presence of network partition, choose between Consistency and Availability. "
                        "Systems can provide at most two of: Consistency, Availability, Partition tolerance."
                    ),
                    concepts=["CAP theorem", "consistency", "availability", "partition tolerance", "trade-off"],
                    hint="During a partition you make a choice; which two can you keep?",
                    tech="general",
                    difficulty="advanced"
                ),
                QuestionDefinition(
                    text="What is the difference between optimistic and pessimistic locking?",
                    answer="Optimistic assumes conflicts rare; checks before commit. Pessimistic locks before modification. Optimistic better for low-contention; pessimistic for high-contention.",
                    concepts=["optimistic locking", "pessimistic locking", "conflict detection", "contention"],
                    hint="Think about when you check for conflicts: before or after?",
                    tech="general",
                    difficulty="advanced"
                ),
                QuestionDefinition(
                    text="How does garbage collection work in memory-managed languages?",
                    answer="Garbage collection frees memory of unreachable objects automatically. Algorithms: mark-and-sweep (marks live, sweeps dead), generational GC (young collected more), reference counting (track references).",
                    concepts=["mark-and-sweep", "generational", "reference counting", "reachability", "memory"],
                    hint="Think about how the system knows which objects are still needed.",
                    tech="general",
                    difficulty="advanced"
                ),
            ],
        },
        "python": {
            "beginner": [
                QuestionDefinition(
                    text="What are lists vs tuples in Python, and when use each?",
                    answer=(
                        "Lists are mutable sequences suitable for items that change. "
                        "Tuples are immutable, often used for fixed collections or as dict keys."
                    ),
                    concepts=["mutable", "immutable", "sequence", "use cases", "dict keys"],
                    hint="One changes, one doesn't; think about when you'd want that.",
                    tech="python",
                    difficulty="beginner"
                ),
            ]
        },
    },
    # Behavioral and scoring rules
    scoring=ScoringRules(
        correct_first_attempt=9,
        partial_then_correct=7,
        partial_then_failed=5,
        incorrect_then_correct=4,
        incorrect_then_failed=1,
        concept_coverage_for_correct=0.5,  # 50% = correct (lenient)
        concept_coverage_for_partial=0.3
    ),
    hinting=HintingRules(),
    retry=RetryRules(),
    feedback=FeedbackRules(),
    
    # Interview structure with dynamic generation enabled
    total_questions=10,  # 4 beginner + 3 intermediate + 3 advanced
    default_tech="general",
    use_dynamic_generation=True,  # ENABLE DYNAMIC QUESTION GENERATION
    difficulty_distribution={
        "beginner": 4,
        "intermediate": 3,
        "advanced": 3
    },
    require_terminal_state=True,
    strict_hint_limit=True,
)
