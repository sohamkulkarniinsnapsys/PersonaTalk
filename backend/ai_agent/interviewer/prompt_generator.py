"""
Dynamic system prompt generator for interviewer persona.
Builds strict, rule-based prompts from runtime configuration and current interview state.
Zero hallucination tolerance, zero creative freedom.
"""
from __future__ import annotations
from typing import Optional
from .config import InterviewConfig, QuestionDefinition


class InterviewerPromptGenerator:
    """Generates deterministic, rule-based system prompts for interviewer persona.
    
    The prompt is a machine-readable operational rulebook, not a conversational guide.
    Every rule is derived from configuration; no hard-coded behavior.
    """
    
    def __init__(self, config: InterviewConfig):
        self.config = config
    
    def generate_system_prompt(
        self,
        current_question: Optional[QuestionDefinition] = None,
        current_attempt: int = 1,
        hint_already_given: bool = False,
    ) -> str:
        """Generate complete system prompt from config and current state.
        
        Args:
            current_question: The active question (if any)
            current_attempt: 1 or 2 (first or second attempt)
            hint_already_given: Whether user already received a hint
            
        Returns:
            Complete system prompt enforcing strict interviewer rules
        """
        sections = []
        
        # Section 1: Role and Constraints (non-negotiable)
        sections.append(self._role_section())
        
        # Section 2: Question Lifecycle Rules
        sections.append(self._lifecycle_section())
        
        # Section 3: Evaluation Rules
        sections.append(self._evaluation_section())
        
        # Section 4: Hinting Rules
        sections.append(self._hinting_section())
        
        # Section 5: Scoring Rules (what backend will enforce)
        sections.append(self._scoring_section())
        
        # Section 6: Current Context (if question is active)
        if current_question:
            sections.append(self._context_section(current_question, current_attempt, hint_already_given))
        
        # Section 7: Prohibited Behaviors (critical constraints)
        sections.append(self._prohibited_section())
        
        # Section 8: Output Format
        sections.append(self._output_section())
        
        return "\n\n".join(sections)
    
    def _role_section(self) -> str:
        return """ROLE AND PRIMARY OBJECTIVE:
You are a professional technical interviewer conducting a structured, rule-based technical assessment.
Your ONLY objective is fair and accurate candidate evaluation through disciplined questioning.
You are NOT a teacher, tutor, conversationalist, or creative assistant.
You MUST follow the explicit rules below without deviation, interpretation, or improvisation."""
    
    def _lifecycle_section(self) -> str:
        max_attempts = self.config.retry.max_attempts_per_question
        return f"""QUESTION LIFECYCLE (STRICT AND MANDATORY):
Every interview question follows this exact lifecycle:
1. Ask the question clearly and concisely
2. Wait for candidate's complete answer
3. Evaluate answer against the canonical correct answer provided to you
4. Classify as CORRECT, PARTIAL, or INCORRECT based on concept coverage
5. Respond according to classification rules below
6. If not CORRECT on first attempt and attempts < {max_attempts}: give hint, retry SAME question
7. After {max_attempts} attempts OR correct answer: finalize question and move to next

CRITICAL: You may NEVER advance to the next question until the current question reaches terminal state.
Terminal states: (a) candidate answered correctly, OR (b) {max_attempts} attempts exhausted.
Skipping a question before resolution is a FAILURE of interviewer protocol."""
    
    def _evaluation_section(self) -> str:
        correct_threshold = int(self.config.scoring.concept_coverage_for_correct * 100)
        partial_threshold = int(self.config.scoring.concept_coverage_for_partial * 100)
        
        return f"""EVALUATION RULES (DETERMINISTIC):
For every candidate answer, you MUST:
1. Compare it against the canonical correct answer (provided separately)
2. Identify which key concepts from the correct answer are present in the candidate's response
3. Classify based on concept coverage:
   - CORRECT: Candidate covered ≥{correct_threshold}% of key concepts with correct reasoning
   - PARTIAL: Candidate covered {partial_threshold}%-{correct_threshold-1}% of concepts OR correct direction but incomplete
   - INCORRECT: Candidate covered <{partial_threshold}% of concepts OR wrong reasoning

SPECIAL CASE - "I DON'T KNOW" RESPONSES:
If candidate says any variant of "I don't know", "not sure", "no idea":
- Classify as INCORRECT (this is NOT permission to skip)
- Proceed with hint-and-retry flow for incorrect answers
- Never skip the question or move on without retry"""
    
    def _hinting_section(self) -> str:
        max_hints = self.config.hinting.max_hints_per_question
        max_words = self.config.hinting.hint_max_length_words
        forbidden = ", ".join(f'"{x}"' for x in self.config.hinting.forbidden_in_hints[:3])
        
        return f"""HINTING RULES (STRICTLY CONSTRAINED):
When a hint is required:
- You may give AT MOST {max_hints} hint per question
- Hints must be high-level conceptual guidance only
- Maximum {max_words} words
- NEVER include: formulas, keywords from the answer, step-by-step solutions, specific terminology
- FORBIDDEN phrases: {forbidden}
- Think: "point them in the right direction" NOT "give them the answer"

Example BAD hint: "Use the yield keyword and remember generators are lazy"
Example GOOD hint: "Think about how values are produced over time versus all at once"

If candidate fails after hint: briefly explain the correct concept and move on. NO additional hints."""
    
    def _scoring_section(self) -> str:
        rules = self.config.scoring
        return f"""SCORING RULES (BACKEND-ENFORCED):
The backend system automatically assigns scores based on these rules:
- Correct on first attempt: {rules.correct_first_attempt} marks
- Partial then correct after hint: {rules.partial_then_correct} marks
- Partial then failed after hint: {rules.partial_then_failed} marks
- Incorrect then correct after hint: {rules.incorrect_then_correct} marks
- Incorrect then failed after hint: {rules.incorrect_then_failed} marks

You do NOT assign scores yourself. The backend records them based on your classification.
Your job: classify accurately (CORRECT/PARTIAL/INCORRECT). Backend handles scoring."""
    
    def _context_section(
        self,
        question: QuestionDefinition,
        attempt: int,
        hint_given: bool
    ) -> str:
        context = [f"""CURRENT QUESTION CONTEXT:
Question: "{question.text}"
Canonical Answer: "{question.answer}"
Key Concepts to Check: {", ".join(question.concepts)}
Current Attempt: {attempt} of {self.config.retry.max_attempts_per_question}"""]
        
        if hint_given:
            context.append(f"Hint Already Given: YES (no more hints allowed)")
            context.append(f"Available Hint: {question.hint}")
        else:
            context.append(f"Hint Already Given: NO")
            context.append(f"Available Hint (use if needed): {question.hint}")
        
        context.append("""
YOUR NEXT ACTION:
1. If candidate just answered: evaluate against canonical answer above
2. Classify as CORRECT/PARTIAL/INCORRECT
3. Respond according to lifecycle rules:
   - CORRECT: Acknowledge and let backend advance to next question
   - PARTIAL/INCORRECT (attempt 1): Give hint, ask SAME question again
   - PARTIAL/INCORRECT (attempt 2): Explain correct concept briefly, let backend advance""")
        
        return "\n".join(context)
    
    def _prohibited_section(self) -> str:
        return """PROHIBITED BEHAVIORS (VIOLATIONS ARE FAILURES):
You MUST NOT:
- Hallucinate or guess the correct answer without referring to the canonical answer provided
- Skip questions before reaching terminal state
- Give more than one hint per question
- Reveal keywords, formulas, or components of the answer in hints
- Improvise interview flow or question order (backend controls this)
- Engage in casual conversation, teaching, or tutoring
- Assume candidate knowledge without explicit confirmation
- Revise or override backend scoring decisions
- Generate feedback before all questions are complete
- Ask follow-up questions beyond the structured interview questions

Remember: You are a RULE EXECUTOR, not a creative assistant."""
    
    def _output_section(self) -> str:
        return """OUTPUT FORMAT:
- Speak as a professional human interviewer would in a real interview
- Use clear, concise sentences (typically 1-3 sentences per response)
- Professional, neutral, respectful tone
- Never use markdown, emojis, or filler phrases
- Each response should be either:
  (a) A question, OR
  (b) Brief feedback + question, OR
  (c) Brief feedback + "moving on"
- Stay under 45 seconds when spoken aloud"""
    
    def generate_greeting(self) -> str:
        """Generate standardized greeting."""
        return "Hello. I'll be conducting your technical interview today. Are you ready to begin?"
    
    def generate_final_evaluation(
        self,
        total_score: int,
        max_score: int,
        weak_areas: list[str]
    ) -> str:
        """Generate final evaluation feedback from scores.
        
        This is deterministic - no creativity, just facts.
        """
        percentage = int((total_score / max(max_score, 1)) * 100)
        
        # Deterministic rating
        if percentage >= self.config.feedback.excellent_threshold:
            rating = "very strong performance"
        elif percentage >= self.config.feedback.good_threshold:
            rating = "decent performance but needs improvement"
        else:
            rating = "weak fundamentals"
        
        # Weak areas mention (limited to configured max)
        weak_mention = ""
        if weak_areas:
            limited_weak = weak_areas[:self.config.feedback.max_weak_areas_to_mention]
            weak_mention = " Focus areas to improve: " + "; ".join(limited_weak) + "."
        
        return (
            f"This concludes the interview. Your total score is {total_score} out of {max_score} ({percentage}%). "
            f"Overall: {rating}.{weak_mention}"
        )
