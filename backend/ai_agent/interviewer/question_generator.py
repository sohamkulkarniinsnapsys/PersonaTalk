"""
Dynamic question generation system for interview personas.

This module generates fresh, topic-based questions at runtime using an LLM,
rather than relying on hard-coded question banks. Questions are generated
on-demand based on selected topics and difficulty levels.
"""
from __future__ import annotations
import logging
import random
from typing import Optional, List, Dict, Any
from .config import QuestionDefinition
from .topic_taxonomy import get_taxonomy, Topic

logger = logging.getLogger(__name__)


class DynamicQuestionGenerator:
    """
    Generates interview questions dynamically at runtime using LLM.
    
    Questions are generated on-demand based on:
    1. Selected domain (Python, JavaScript, React, etc.)
    2. Current difficulty level (beginner, intermediate, advanced)
    3. Selected topic within that domain/difficulty
    
    This ensures:
    - No hard-coded questions (fresh every session)
    - Variety across interviews (different aspects of same topic)
    - Grounded in topic/difficulty (no creative freedom)
    - Deduplication possible (track generated questions across sessions)
    """

    def __init__(self, orchestrator=None, session_seed: Optional[int] = None):
        """
        Initialize the dynamic question generator.
        
        Args:
            orchestrator: LLM orchestrator for generating questions
            session_seed: Optional seed for reproducibility within a session
        """
        self.orchestrator = orchestrator
        self.taxonomy = get_taxonomy()
        self.session_seed = session_seed or random.randint(0, 999999)
        self._generated_count = 0
        logger.info(f"✅ Initialized DynamicQuestionGenerator with session_seed={self.session_seed}")

    def select_random_topics(
        self,
        domain_slug: str,
        count: int = 10,
        distribution: Optional[Dict[str, int]] = None
    ) -> List[Topic]:
        """
        Select random topics from a domain across all difficulty levels.
        
        Args:
            domain_slug: Domain to select from (e.g., "python", "javascript")
            count: Total number of topics to select
            distribution: Optional override for difficulty distribution
                         (default: {beginner: 4, intermediate: 3, advanced: 3})
        
        Returns:
            List of selected Topic objects, balanced by difficulty
        """
        if distribution is None:
            distribution = self.taxonomy.get_difficulty_distribution()
        
        domain = self.taxonomy.get_domain(domain_slug)
        if not domain:
            logger.error(f"❌ Domain '{domain_slug}' not found. Using 'general' fallback.")
            domain = self.taxonomy.get_domain("general")
        
        selected_topics = []
        
        for difficulty in ["beginner", "intermediate", "advanced"]:
            needed = distribution.get(difficulty, 0)
            available = self.taxonomy.get_topics_for_domain_difficulty(domain_slug, difficulty)
            
            if not available:
                logger.warning(f"⚠️ No {difficulty} topics found for domain '{domain_slug}'")
                continue
            
            # Set seed for reproducibility within this session
            random.seed(self.session_seed + hash(difficulty) % 10000)
            selected = random.sample(available, min(needed, len(available)))
            selected_topics.extend(selected)
            logger.info(f"   📚 Selected {len(selected)}/{needed} {difficulty} topics")
        
        logger.info(f"✅ Selected {len(selected_topics)} topics total")
        return selected_topics

    async def generate_question(
        self,
        topic: Topic,
        context: Optional[str] = None
    ) -> QuestionDefinition:
        """
        Generate a single interview question for a given topic.
        
        This uses the LLM to create a fresh question based on the topic,
        difficulty, and key concepts. The question is deterministic but
        varied (different phrasing/angle each time).
        
        Args:
            topic: The Topic object specifying what to ask about
            context: Optional additional context (e.g., interview history)
        
        Returns:
            QuestionDefinition with generated question, answer, and concepts
        """
        if not self.orchestrator:
            logger.error("❌ No orchestrator provided; cannot generate questions")
            raise ValueError("Orchestrator required for question generation")
        
        self._generated_count += 1
        
        # Build prompt for LLM to generate question
        generation_prompt = self._build_generation_prompt(topic, context)
        
        messages = [
            {
                "role": "system",
                "content": self._generation_system_prompt()
            },
            {
                "role": "user",
                "content": generation_prompt
            }
        ]
        
        try:
            # Request LLM to generate question as JSON
            llm_response = await self.orchestrator.llm.generate_response(
                messages,
                system_prompt=self._generation_system_prompt()
            )
            
            response_text = llm_response.get('text', '')
            
            # Parse JSON response to QuestionDefinition
            import json
            try:
                # Try to extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    question_data = json.loads(json_str)
                    
                    qdef = QuestionDefinition(
                        text=question_data.get('question', 'Question generation failed'),
                        answer=question_data.get('answer', ''),
                        concepts=question_data.get('concepts', []),
                        hint=question_data.get('hint', 'Think about the core concepts.'),
                        tech=getattr(topic, "domain", None) or "general",
                        difficulty=topic.difficulty
                    )
                    logger.info(f"✅ Generated {topic.difficulty} question #{self._generated_count} on '{topic.name}'")
                    return qdef
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️ Failed to parse LLM JSON response: {e}")
        
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}", exc_info=True)
        
        # Fallback: Create a generic question if generation fails
        logger.warning(f"⚠️ Falling back to generic question for topic '{topic.name}'")
        return self._fallback_question(topic)

    def _build_generation_prompt(self, topic: Topic, context: Optional[str]) -> str:
        """Build the prompt to send to LLM for question generation."""
        prompt = f"""Generate a {topic.difficulty.upper()} level technical interview question about: {topic.name}

Topic Description: {topic.description}
Key Concepts: {', '.join(topic.key_concepts)}

Requirements:
1. Question should test understanding of: {', '.join(topic.key_concepts[:5])}
2. Difficulty level: {topic.difficulty}
3. Answer should demonstrate understanding, not just memorization
4. Include a conceptual hint that doesn't reveal the answer

Respond ONLY with valid JSON in this exact format:
{{
    "question": "Clear, specific question text",
    "answer": "Comprehensive correct answer",
    "concepts": ["concept1", "concept2", "concept3"],
    "hint": "Conceptual hint pointing in right direction"
}}

Generate now:"""
        
        if context:
            prompt = f"{context}\n\n{prompt}"
        
        return prompt

    def _generation_system_prompt(self) -> str:
        """System prompt for the LLM during question generation."""
        return """You are an expert technical interview question generator. 

Your task is to create high-quality, fair, and clear technical interview questions.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no code blocks, no explanation
2. Questions must be specific and measurable
3. Answers must be comprehensive but concise
4. Concepts must be 3-5 key terms that would appear in a correct answer
5. Hints must be conceptual guidance only - never reveal keywords or formulas
6. Ensure question can be answered by someone with topic expertise
7. Make questions original and distinct from typical pattern questions

Example JSON format (you MUST follow this exactly):
{
    "question": "What is the primary difference between let and var in JavaScript scoping?",
    "answer": "let is block-scoped while var is function-scoped. let is not hoisted to the top of its scope, var is. let cannot be redeclared in the same scope, var can.",
    "concepts": ["block scope", "function scope", "hoisting", "redeclaration"],
    "hint": "Think about which scope (block vs function) each declaration respects, and what happens at the top of the scope."
}

Remember: Output ONLY the JSON object. Start with { and end with }."""

    def _fallback_question(self, topic: Topic) -> QuestionDefinition:
        """Create a fallback generic question if LLM generation fails."""
        tech = getattr(topic, "domain", None) or "general"
        return QuestionDefinition(
            text=f"Explain the key concepts of {topic.name}.",
            answer="Please provide an explanation covering the main concepts and their relationships.",
            concepts=topic.key_concepts[:5],
            hint=f"Think about {', '.join(topic.key_concepts[:2])} and how they relate.",
            tech=tech,
            difficulty=topic.difficulty
        )


class QuestionDeduplicator:
    """
    Tracks and deduplicates generated questions across interview sessions.
    
    Maintains a history of generated questions to prevent asking the same
    question multiple times within a reasonable window (e.g., per user, per day).
    """

    def __init__(self):
        """Initialize the deduplicator with empty history."""
        self.generated_questions: Dict[str, List[str]] = {}  # domain -> list of question texts
        logger.info("✅ Initialized QuestionDeduplicator")

    def has_been_asked(self, domain: str, question_text: str) -> bool:
        """Check if a question has been asked recently in this domain."""
        if domain not in self.generated_questions:
            return False
        
        # Simple string similarity check (for exact duplicates)
        normalized_q = question_text.lower().strip()
        for prev_q in self.generated_questions.get(domain, []):
            if prev_q.lower().strip() == normalized_q:
                return True
        
        return False

    def record_question(self, domain: str, question_text: str):
        """Record that a question was asked in a domain."""
        if domain not in self.generated_questions:
            self.generated_questions[domain] = []
        
        self.generated_questions[domain].append(question_text)
        
        # Keep history to reasonable size (last 100 questions per domain)
        if len(self.generated_questions[domain]) > 100:
            self.generated_questions[domain] = self.generated_questions[domain][-100:]

    def clear_history(self, domain: Optional[str] = None):
        """Clear question history."""
        if domain:
            if domain in self.generated_questions:
                del self.generated_questions[domain]
        else:
            self.generated_questions.clear()
