from __future__ import annotations
import asyncio
import logging
import random
from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async

from ai_agent.models import Call, InterviewSession, InterviewQuestionState
from ai_agent.conversation import ConversationPhase, ConversationSession
from .config import InterviewConfig, QuestionDefinition, DEFAULT_INTERVIEWER_CONFIG
from .prompt_generator import InterviewerPromptGenerator
from .evaluator import classify, score_for, Classification
from .question_generator import DynamicQuestionGenerator, QuestionDeduplicator
from .topic_taxonomy import get_taxonomy

logger = logging.getLogger(__name__)


class InterviewerController:
    """Strict, isolated interviewer persona controller.

    - Enforces per-question lifecycle with two attempts max
    - Persists state to DB (InterviewSession + InterviewQuestionState)
    - Never advances until terminal resolution
    - Provides final evaluation and feedback at the end
    """

    def __init__(self, room_id: str, call_id: int, persona_slug: str, persona_config: Dict[str, Any], orchestrator, send_audio_callable, notify_callback=None, session: ConversationSession | None = None):
        self.room_id = room_id
        self.call_id = call_id
        self.persona_slug = persona_slug
        self.persona_config = persona_config
        self.orchestrator = orchestrator
        self.send_audio = send_audio_callable
        self._notify_callback = notify_callback
        self._lock = asyncio.Lock()
        self._greeting_sent = False
        self._tech: str | None = None
        self._stage: str = "beginner"  # Current difficulty: beginner/intermediate/advanced
        self._question_index: int = 0
        self._questions_per_difficulty = {"beginner": 0, "intermediate": 0, "advanced": 0}  # Track count per level
        self._difficulty_limits = {"beginner": 4, "intermediate": 3, "advanced": 3}  # Standard distribution
        self.session: ConversationSession | None = session
        self._ai_turn_seq: int = 0
        self._utterance_seq: int = 0
        
        # Conversation history for LLM context (role/text pairs)
        self._history: list[dict[str, str]] = []
        
        # Load dynamic config from persona metadata or use defaults
        config_data = self.persona_config.get("metadata", {}).get("interview_config", {})
        if config_data:
            self.config = InterviewConfig.from_dict(config_data)
        else:
            self.config = DEFAULT_INTERVIEWER_CONFIG
        
        # Initialize dynamic prompt generator
        self.prompt_gen = InterviewerPromptGenerator(self.config)
        
        # Initialize dynamic question generator (uses LLM to generate questions at runtime)
        session_seed = hash(self.room_id) % 1000000
        self.question_generator = DynamicQuestionGenerator(
            orchestrator=orchestrator,
            session_seed=session_seed
        )
        
        # Initialize question deduplicator to track generated questions
        self.question_dedup = QuestionDeduplicator()
        
        # Pre-selected topics for this interview (selected on first user input)
        self._selected_topics: list = []
        
        # Cache of generated questions for this interview
        self._generated_questions_cache: Dict[str, QuestionDefinition] = {}

        # Prime WAIT_FOR_USER so VAD does not drop the first utterance
        if self.session:
            self.session.phase = ConversationPhase.WAIT_FOR_USER
            self.session.expected_user_utterance_id = self._next_utterance_id()
        
        logger.info(
            f"✅ Initialized interviewer (room={room_id}, "
            f"dynamic_gen={self.config.use_dynamic_generation}, "
            f"total_questions={self.config.total_questions})"
        )

    # --------- Shared helpers (IDs, notifications) ---------
    def _next_ai_turn_id(self) -> str:
        self._ai_turn_seq += 1
        return f"ai-{self._ai_turn_seq}"

    def _next_utterance_id(self) -> str:
        self._utterance_seq += 1
        return f"utt-{self._utterance_seq}"

    async def _emit_transcript(self, role: str, text: str, turn_id: str | None = None, utterance_id: str | None = None):
        if not self._notify_callback or not text:
            return
        try:
            import time

            await self._notify_callback({
                "type": "transcript",
                "role": role,
                "text": text,
                "turnId": turn_id,
                "utteranceId": utterance_id,
                "roomId": self.room_id,
                "timestamp": time.time(),
            })
        except Exception:
            logger.debug("notify_callback failed for transcript", exc_info=True)

    async def _notify_state(self, state: ConversationPhase):
        if self.session:
            self.session.phase = state
        if not self._notify_callback:
            return
        try:
            import time

            await self._notify_callback({
                "type": "agent_state",
                "state": state.value if isinstance(state, ConversationPhase) else str(state),
                "roomId": self.room_id,
                "timestamp": time.time(),
            })
        except Exception:
            logger.debug("notify_callback failed for agent_state", exc_info=True)

    async def _set_waiting_for_user(self, mark_greeting_played: bool = False):
        if self.session:
            self.session.active_user_utterance_id = None
            self.session.expected_user_utterance_id = self._next_utterance_id()
            self.session.phase = ConversationPhase.WAIT_FOR_USER
            if mark_greeting_played:
                self.session.greeting_sent_and_played = True
                self.session.started = True
        await self._notify_state(ConversationPhase.WAIT_FOR_USER)

    # --------- Persistence helpers ---------
    @sync_to_async
    def _get_or_create_session(self) -> InterviewSession:
        call = Call.objects.get(pk=self.call_id)
        sess, _ = InterviewSession.objects.get_or_create(call=call, persona_slug=self.persona_slug)
        return sess

    @sync_to_async
    def _current_q(self, sess: InterviewSession) -> Optional[InterviewQuestionState]:
        try:
            return sess.questions.get(index=self._question_index)
        except InterviewQuestionState.DoesNotExist:
            return None

    @sync_to_async
    def _create_q_from_def(self, sess: InterviewSession, qdef: QuestionDefinition):
        return InterviewQuestionState.objects.create(
            session=sess,
            index=self._question_index,
            tech=qdef.tech,
            stage=qdef.difficulty,
            question_text=qdef.text,
            reference_answer=qdef.answer,
            key_concepts=qdef.concepts,
        )

    @sync_to_async
    def _save_first_attempt(self, q: InterviewQuestionState, user_text: str, label: str, hint_given: bool):
        q.first_response = user_text
        q.first_evaluation = label
        q.hint_given = hint_given
        q.save(update_fields=["first_response", "first_evaluation", "hint_given", "updated_at"])

    @sync_to_async
    def _save_second_attempt(self, q: InterviewQuestionState, user_text: str, final_score: int, feedback: str, derivation: Dict[str, Any]):
        q.second_response = user_text
        q.final_score = final_score
        q.finalized = True
        q.feedback = feedback
        q.derivation = derivation
        q.save(update_fields=["second_response", "final_score", "finalized", "feedback", "derivation", "updated_at"])

    @sync_to_async
    def _finalize_first_attempt_correct(self, q: InterviewQuestionState, final_score: int, feedback: str, derivation: Dict[str, Any]):
        q.final_score = final_score
        q.finalized = True
        q.feedback = feedback
        q.derivation = derivation
        q.save(update_fields=["final_score", "finalized", "feedback", "derivation", "updated_at"])

    @sync_to_async
    def _advance_index(self):
        # Increment count for current difficulty
        self._questions_per_difficulty[self._stage] += 1
        logger.info(
            f"✅ Completed {self._stage} question. Progress: "
            f"beginner={self._questions_per_difficulty['beginner']}/{self._difficulty_limits['beginner']}, "
            f"intermediate={self._questions_per_difficulty['intermediate']}/{self._difficulty_limits['intermediate']}, "
            f"advanced={self._questions_per_difficulty['advanced']}/{self._difficulty_limits['advanced']}"
        )
        self._question_index += 1
        
        # Check if current difficulty level is complete - move to next difficulty
        if self._questions_per_difficulty[self._stage] >= self._difficulty_limits[self._stage]:
            if self._stage == "beginner":
                self._stage = "intermediate"
                logger.info(f"📈 Progressing to INTERMEDIATE difficulty (completed {self._questions_per_difficulty['beginner']} beginner questions)")
            elif self._stage == "intermediate":
                self._stage = "advanced"
                logger.info(f"📈 Progressing to ADVANCED difficulty (completed {self._questions_per_difficulty['intermediate']} intermediate questions)")

    # --------- Dynamic Question Generation ---------
    async def _generate_next_dynamic_question(self) -> Optional[QuestionDefinition]:
        """Generate the next interview question dynamically using LLM."""
        try:
            # On first question, select topics for the entire interview
            if not self._selected_topics:
                logger.info(f"🎲 Selecting topics for {self._stage} difficulty...")
                domain = self._tech or self.config.default_tech
                self._selected_topics = self.question_generator.select_random_topics(
                    domain_slug=domain,
                    count=self.config.total_questions,
                    distribution=self.config.difficulty_distribution
                )
                if not self._selected_topics:
                    logger.error(f"❌ No topics selected for domain '{domain}'")
                    return None
            
            # Get next topic for current difficulty
            current_difficulty_limit = self._difficulty_limits[self._stage]
            current_difficulty_count = self._questions_per_difficulty[self._stage]
            
            # Filter topics for current difficulty and pick next
            topics_for_difficulty = [
                t for t in self._selected_topics 
                if t.difficulty == self._stage
            ]
            
            if not topics_for_difficulty:
                logger.error(f"❌ No topics found for difficulty '{self._stage}'")
                return None
            
            # Cycle through topics within difficulty (index within that difficulty level)
            topic_idx = current_difficulty_count % len(topics_for_difficulty)
            selected_topic = topics_for_difficulty[topic_idx]
            
            logger.info(f"📚 Generating {self._stage} question on topic: {selected_topic.name}")
            
            # Generate fresh question for this topic
            qdef = await self.question_generator.generate_question(selected_topic)
            
            # Track generated question for deduplication
            self.question_dedup.record_question(self._tech or "general", qdef.text)
            
            # Cache for later reference
            cache_key = f"{self._stage}_{current_difficulty_count}"
            self._generated_questions_cache[cache_key] = qdef
            
            return qdef
            
        except Exception as e:
            logger.error(f"❌ Dynamic question generation failed: {e}", exc_info=True)
            return None

    def _get_static_question(self) -> Optional[QuestionDefinition]:
        """Fallback: Get question from static question bank."""
        tech = self._tech or self.config.default_tech
        tech_bank = self.config.questions.get(tech, self.config.questions.get(self.config.default_tech, {}))
        stage_set = tech_bank.get(self._stage, tech_bank.get("beginner", []))
        
        if not stage_set:
            logger.warning(f"⚠️ No static questions found for tech={tech}, difficulty={self._stage}")
            return None
        
        # Shuffle for variety using session seed
        session_seed = hash(self.room_id + self._stage) % 10000
        random.seed(session_seed)
        shuffled = list(stage_set)
        random.shuffle(shuffled)
        
        # Pick based on offset
        offset = self._questions_per_difficulty[self._stage]
        idx = offset % len(shuffled)
        return shuffled[idx]

    def _get_fallback_question(self) -> QuestionDefinition:
        """Generic fallback question."""
        difficulty = self._stage
        return QuestionDefinition(
            text=f"Explain a key concept related to {self._tech or 'software development'}.",
            answer="Please provide a clear, detailed explanation demonstrating your understanding.",
            concepts=["understanding", "explanation", "concepts"],
            hint="Think about the fundamental ideas and how they work together.",
            tech=self._tech or "general",
            difficulty=difficulty
        )

    @sync_to_async
    def _update_total_score(self, sess: InterviewSession):
        scores = list(sess.questions.filter(finalized=True).values_list("final_score", flat=True))
        sess.total_score = float(sum(s for s in scores if s is not None))
        sess.save(update_fields=["total_score", "updated_at"])

    # --------- Public API ---------
    async def start(self):
        async with self._lock:
            if self._greeting_sent:
                return
            
            # CRITICAL: Greeting should ask about technology, NOT ask technical questions
            greeting = "Good morning. Thank you for joining today's technical interview. To begin, which technology or programming language do you primarily work with? For example, Python, JavaScript, Java, or general software concepts?"
            
            # Add to history
            self._history.append({"role": "assistant", "text": greeting})
            ai_turn_id = self._next_ai_turn_id()
            if self.session:
                self.session.last_ai_turn_id = ai_turn_id
            await self._speak(greeting, turn_id=ai_turn_id, mark_greeting_played=True)
            self._greeting_sent = True
            logger.info("🎤 Interviewer greeted and asked about technology preference")

    async def handle_user_utterance(self, text: str, utterance_id: Optional[str] = None):
        async with self._lock:
            # Add user input to history FIRST
            self._history.append({"role": "user", "text": text})
            
            # Tech selection step (first user turn): pick tech from user's words
            if self._tech is None:
                logger.info(f"🤖 Detecting technology from first user input: '{text}'")
                self._tech = self._detect_tech(text)
                logger.info(f"   ✅ Detected tech: {self._tech}")
                await self._ask_current_question(force_new=True)
                return

            # We must have an active question object
            sess = await self._get_or_create_session()
            q = await self._current_q(sess)
            if q is None:
                logger.warning(f"⚠️ No question at index {self._question_index}; creating new one")
                await self._ask_current_question(force_new=True)
                return

            # "I don't know" strict handling
            if self._is_idk(text):
                logger.info(f"🤔 User said 'I don't know'. Treating as incorrect on current question.")
                await self._handle_idk(q)
                return

            # Evaluate first or second attempt based on database state
            if not q.first_response:
                logger.info(f"📝 Processing FIRST attempt for question index={self._question_index}")
                await self._evaluate_first_attempt(q, text)
                return  # Explicit return after first attempt
            elif not q.finalized:
                logger.info(f"📝 Processing SECOND attempt for question index={self._question_index}")
                await self._evaluate_second_attempt(q, text)
                return  # Explicit return after second attempt
            else:
                # If finalized unexpectedly, something went wrong - ask next question
                logger.warning(f"⚠️ Question at index {self._question_index} already finalized; moving to next")
                await self._ask_current_question(force_new=True)
                return

    # --------- Core behavior ---------
    async def _ask_current_question(self, force_new: bool = False):
        sess = await self._get_or_create_session()
        q = await self._current_q(sess)

        if q is None or force_new:
            # Check if all difficulties are complete (interview finished)
            total_completed = sum(self._questions_per_difficulty.values())
            total_limit = sum(self._difficulty_limits.values())
            
            if total_completed >= total_limit:
                logger.info("✅ All questions completed - triggering final evaluation")
                await self._final_evaluation()
                return
            
            # DYNAMIC GENERATION: Generate fresh question at runtime
            if self.config.use_dynamic_generation:
                qdef = await self._generate_next_dynamic_question()
                if qdef is None:
                    logger.error("❌ Dynamic question generation failed, using fallback")
                    qdef = self._get_fallback_question()
            else:
                # Fallback: Use pre-defined question bank
                qdef = self._get_static_question()
            
            if qdef is None:
                logger.error("❌ No question available; cannot continue")
                await self._speak("I apologize, but we've encountered a configuration issue. Let's end the interview here.")
                return
            
            q = await self._create_q_from_def(sess, qdef)
            question_num = total_completed + 1
            logger.info(f"❓ Asking question #{question_num}/{total_limit}: {self._stage.upper()} - {qdef.text[:60]}...")
        
        # Generate dynamic system prompt with current question context
        system_prompt = self.prompt_gen.generate_system_prompt(
            current_question=self._qdef_from_db_question(q),
            current_attempt=1 if not q.first_response else 2,
            hint_already_given=q.hint_given
        )
        
        # Ask LLM to present the question naturally
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": m["role"], "content": m["text"]} for m in self._history])
        messages.append({"role": "user", "content": f"Present the question: {q.question_text}"})
        
        try:
            llm_response = await self.orchestrator.llm.generate_response(messages, system_prompt)
            response_text = llm_response.get('text', q.question_text)
            
            # Add to history
            self._history.append({"role": "assistant", "text": response_text})
            
            await self._speak(response_text)
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}, using fallback")
            self._history.append({"role": "assistant", "text": q.question_text})
            await self._speak(q.question_text)

    async def _evaluate_first_attempt(self, q: InterviewQuestionState, user_text: str):
        # Use config-driven evaluation thresholds
        verdict = classify(
            user_text,
            q.reference_answer,
            q.key_concepts,
            correct_threshold=self.config.scoring.concept_coverage_for_correct,
            partial_threshold=self.config.scoring.concept_coverage_for_partial
        )
        label = verdict["label"]
        logger.info(f"   📊 First attempt verdict: {label} (user: '{user_text[:60]}...')")

        if label == Classification.CORRECT:
            logger.info(f"   ✅ CORRECT on first attempt! Finalizing and advancing.")
            score, why = score_for(
                Classification.CORRECT,
                None,
                scoring_rules=self.config.scoring
            )
            await self._finalize_first_attempt_correct(q, score, "Correct", {"rule": why})
            
            # Get LLM to generate natural acknowledgment
            response_text = await self._get_llm_response(
                q, 
                current_attempt=1,
                hint_already_given=q.hint_given,
                instruction="Acknowledge the correct answer briefly and indicate you're moving to the next question."
            )
            
            self._history.append({"role": "assistant", "text": response_text})
            await self._speak(response_text)
            
            await self._advance_index()
            await self._update_total_score(await self._get_or_create_session())
            if await self._maybe_finalize():
                return
            await self._ask_current_question(force_new=True)
            return

        # Partial or incorrect → give ONE high-level hint and retry SAME question
        logger.info(f"   ❌ {label.upper()} on first attempt. Giving hint and waiting for retry on SAME question.")
        hint = self._high_level_hint(q)
        await self._save_first_attempt(q, user_text, label, True)
        
        # Get LLM to generate natural hint + retry request
        instruction = (
            f"The answer was {label}. "
            f"Provide this hint: {hint}. "
            "Then ask the candidate to try answering the SAME question again. "
            "Do NOT move to the next question. Do NOT reveal the answer."
        )
        
        response_text = await self._get_llm_response(
            q,
            current_attempt=1,
            hint_already_given=False,  # Before we give the hint
            instruction=instruction
        )
        
        self._history.append({"role": "assistant", "text": response_text})
        await self._speak(response_text)
        # CRITICAL: Return here - do NOT call _advance_index() or _ask_current_question()
        logger.info(f"   🎤 Hint given. Waiting for user's second attempt on same question (index={self._question_index})")
        return

    async def _evaluate_second_attempt(self, q: InterviewQuestionState, user_text: str):
        logger.info(f"   📊 Evaluating SECOND attempt for question (index={self._question_index})")
        first_label = q.first_evaluation or Classification.INCORRECT
        verdict = classify(
            user_text,
            q.reference_answer,
            q.key_concepts,
            correct_threshold=self.config.scoring.concept_coverage_for_correct,
            partial_threshold=self.config.scoring.concept_coverage_for_partial
        )
        second_label = verdict["label"]
        logger.info(f"   📊 Second attempt result: {second_label} (first was {first_label})")
        score, why = score_for(first_label, second_label, scoring_rules=self.config.scoring)

        # Generate natural feedback using LLM
        if second_label == Classification.CORRECT:
            instruction = "Acknowledge the improvement and indicate you're moving to the next question."
        else:
            # Reveal key concepts without full answer
            concepts = ", ".join(q.key_concepts[:3]) if q.key_concepts else "the core concepts"
            instruction = (
                f"The answer was still {second_label} after the hint. "
                f"Briefly mention these key concepts: {concepts}. "
                "Then indicate you're moving to the next question."
            )
        
        response_text = await self._get_llm_response(
            q,
            current_attempt=2,
            hint_already_given=q.hint_given,
            instruction=instruction
        )
        
        # Finalize and move on
        await self._save_second_attempt(q, user_text, score, response_text, {"rule": why})
        self._history.append({"role": "assistant", "text": response_text})
        await self._speak(response_text)
        
        # NOW advance and move to next question
        logger.info(f"   ✅ Question finalized. Advancing from index {self._question_index} to next.")
        await self._advance_index()
        await self._update_total_score(await self._get_or_create_session())
        if await self._maybe_finalize():
            return
        logger.info(f"   ❓ Now asking next question at index {self._question_index}")
        await self._ask_current_question(force_new=True)
        return

    async def _handle_idk(self, q: InterviewQuestionState):
        # Treat as incorrect, stay on SAME question, give conceptual hint only
        hint = self._high_level_hint(q)
        if not q.first_response:
            await self._save_first_attempt(q, "I don't know", Classification.INCORRECT, True)
        
        # Get LLM to deliver hint naturally while enforcing retry
        instruction = (
            "The candidate said 'I don't know'. "
            f"Provide this conceptual hint: {hint}. "
            "Then ask them to try answering the SAME question again. "
            "Do NOT move to the next question. Do NOT reveal the answer. "
            "Emphasize we'll stay on this question until they attempt an answer."
        )
        
        response_text = await self._get_llm_response(
            q,
            current_attempt=1 if not q.first_response else 2,
            hint_already_given=False,  # About to give hint
            instruction=instruction
        )
        
        self._history.append({"role": "assistant", "text": response_text})
        await self._speak(response_text)

    # --------- Utils ---------
    async def handle_interruption(self, text: str, utterance_id: Optional[str] = None):
        # Treat interruptions as normal answers for interviewer; do not advance phases specially
        await self.handle_user_utterance(text, utterance_id)

    async def _speak(self, text: str, turn_id: str | None = None, mark_greeting_played: bool = False):
        try:
            if not turn_id:
                turn_id = self._next_ai_turn_id()
            if self.session:
                self.session.last_ai_turn_id = turn_id

            await self._notify_state(ConversationPhase.AI_SPEAKING)
            await self._emit_transcript("assistant", text, turn_id=turn_id)

            audio = await self.orchestrator.tts.synthesize(text, self.persona_config.get("voice", {}))
            asyncio.ensure_future(self.send_audio(audio, turn_id))
        except Exception as e:
            logger.error(f"TTS failed: {e}")

        await self._set_waiting_for_user(mark_greeting_played=mark_greeting_played)

    def _detect_tech(self, text: str) -> str:
        """Detect tech from user input. Extensible via config."""
        s = (text or "").lower()
        
        # Tech detection rules (could be config-driven in future)
        tech_keywords = {
            "javascript": ["javascript", "js", "react", "next", "node"],
            "python": ["python", "django", "flask", "fastapi"],
        }
        
        for tech, keywords in tech_keywords.items():
            if tech in self.config.questions and any(k in s for k in keywords):
                return tech
        
        return self.config.default_tech

    def _is_idk(self, text: str) -> bool:
        """Check if answer is an 'I don't know' variant. Uses config."""
        s = (text or "").lower().strip()
        return any(keyword in s for keyword in self.config.retry.idk_keywords)

    def _high_level_hint(self, q: InterviewQuestionState) -> str:
        """Get high-level conceptual hint from config. Enforces hinting rules."""
        # Try to find the question definition in config to get the hint
        tech_bank = self.config.questions.get(q.tech, {})
        for difficulty_level, questions in tech_bank.items():
            for qdef in questions:
                if qdef.text == q.question_text:
                    hint = qdef.hint
                    # Validate hint against config rules
                    if hint and not any(forbidden in hint.lower() for forbidden in self.config.hinting.forbidden_in_hints):
                        return hint
        
        # Fallback to generic concept reminder
        return "Think about the fundamental concepts and how they relate."

    def _final_feedback(self, second_label: str, q: InterviewQuestionState) -> str:
        """Generate feedback for second attempt outcome."""
        if second_label == Classification.CORRECT:
            return "Good improvement. Let's move on."
        else:
            # Provide brief correct concept without full answer
            concepts = ", ".join(q.key_concepts[:3]) if q.key_concepts else "the core concepts"
            return f"The key ideas were: {concepts}. Let's continue."
    
    async def _get_llm_response(
        self,
        q: InterviewQuestionState,
        current_attempt: int,
        hint_already_given: bool,
        instruction: str
    ) -> str:
        """Get natural LLM response with strict system prompt enforcing rules."""
        # Generate dynamic system prompt with current question context
        qdef = self._qdef_from_db_question(q)
        system_prompt = self.prompt_gen.generate_system_prompt(
            current_question=qdef,
            current_attempt=current_attempt,
            hint_already_given=hint_already_given
        )
        
        # Build messages: system prompt + history + instruction
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": m["role"], "content": m["text"]} for m in self._history])
        messages.append({"role": "user", "content": instruction})
        
        try:
            llm_response = await self.orchestrator.llm.generate_response(messages, system_prompt)
            return llm_response.get('text', "Moving on.")
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}, using fallback")
            return "Let's continue."
    
    def _qdef_from_db_question(self, q: InterviewQuestionState) -> QuestionDefinition:
        """Reconstruct QuestionDefinition from database question state."""
        # Find matching question definition in config
        tech_bank = self.config.questions.get(q.tech, {})
        for difficulty, questions in tech_bank.items():
            for qdef in questions:
                if qdef.text == q.question_text:
                    return qdef
        
        # Fallback: reconstruct from DB fields
        from .config import QuestionDefinition
        return QuestionDefinition(
            text=q.question_text,
            answer=q.reference_answer,
            concepts=q.key_concepts or [],
            hint="Consider the fundamental concepts.",
            tech=q.tech,
            difficulty=q.stage
        )

    # --------- Final evaluation ---------
    @sync_to_async
    def _compute_summary(self) -> Dict[str, Any]:
        sess = InterviewSession.objects.get(call_id=self.call_id, persona_slug=self.persona_slug)
        qs = list(sess.questions.all())
        total_possible = max(1, len(qs) * 10)
        got = sum(q.final_score or 0 for q in qs)
        pct = int(round((got / total_possible) * 100))
        weak = [q.question_text for q in qs if (q.final_score or 0) <= 4]
        return {"score": got, "total": total_possible, "pct": pct, "weak": weak}

    async def _maybe_finalize(self) -> bool:
        # Check both: question index vs configured total, and per-difficulty limits
        total_completed = sum(self._questions_per_difficulty.values())
        total_limit = sum(self._difficulty_limits.values())

        if self._question_index < self.config.total_questions and total_completed < total_limit:
            return False

        # Finalize interview
        summary = await self._compute_summary()
        final_text = self.prompt_gen.generate_final_evaluation(
            total_score=summary["score"],
            max_score=summary["total"],
            weak_areas=summary["weak"]
        )

        await self._speak(final_text)
        try:
            sess = await self._get_or_create_session()
            sess.completed = True
            await sync_to_async(sess.save)(update_fields=["completed", "updated_at"])
        except Exception:
            pass
        return True

    async def _final_evaluation(self) -> None:
        """Run final evaluation and mark interview complete.

        This mirrors the logic in `_maybe_finalize` and is invoked when
        `_ask_current_question` detects all difficulty buckets are complete.
        """
        summary = await self._compute_summary()
        final_text = self.prompt_gen.generate_final_evaluation(
            total_score=summary["score"],
            max_score=summary["total"],
            weak_areas=summary["weak"]
        )

        await self._speak(final_text)
        try:
            sess = await self._get_or_create_session()
            sess.completed = True
            await sync_to_async(sess.save)(update_fields=["completed", "updated_at"])
        except Exception:
            logger.debug("Failed to mark session completed", exc_info=True)
