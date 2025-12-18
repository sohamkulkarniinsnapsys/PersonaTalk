import enum
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lightweight, deterministic interview question bank
# Each item: {"q": str, "keywords": [..], "hint": str}
QUESTION_BANK: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "general": {
        "basic": [
            {"q": "In simple terms, what is a REST API and how is it different from RPC?", "keywords": ["rest", "http", "resource", "rpc"], "hint": "Think about HTTP verbs and resources versus function calls."},
            {"q": "What do HTTP 200 and 404 status codes mean?", "keywords": ["200", "ok", "success", "404", "not", "found"], "hint": "200 is success; 404 means the resource is missing."},
            {"q": "What is the time complexity of binary search and why?", "keywords": ["log", "logarithmic", "halve"], "hint": "You cut the search space in half each step."},
        ],
        "moderate": [
            {"q": "How would you design a rate limiter for an API? Mention one algorithm.", "keywords": ["token", "bucket", "leaky", "redis", "window"], "hint": "Token bucket or leaky bucket with a shared store like Redis works well."},
            {"q": "Can you summarize the ACID properties for databases?", "keywords": ["atomic", "consist", "isolation", "durable"], "hint": "Atomicity, Consistency, Isolation, Durability."},
            {"q": "How does DNS resolution work at a high level?", "keywords": ["dns", "resolver", "cache", "recursive", "authoritative"], "hint": "Client -> resolver -> recursive lookups -> authoritative server, with caching."},
        ],
        "advanced": [
            {"q": "Design a scalable real-time chat service. What building blocks would you use?", "keywords": ["websocket", "pub", "sub", "queue", "shard", "presence"], "hint": "WebSockets plus pub/sub (Redis/Kafka), sharding channels, and presence tracking."},
            {"q": "Explain the CAP theorem trade-offs for distributed systems.", "keywords": ["consistency", "availability", "partition"], "hint": "During partition you choose between Consistency or Availability."},
            {"q": "How would you diagnose high tail latency in a service?", "keywords": ["tracing", "profiling", "p99", "queue", "contention"], "hint": "Use tracing/profiling, look for queuing, locks, GC, and slow dependencies."},
        ],
    },
    "javascript": {
        "basic": [
            {"q": "What is the difference between var, let, and const in JavaScript?", "keywords": ["scope", "hoist", "block", "reassign"], "hint": "Think about block scope and reassignment rules."},
            {"q": "Explain how promises help with async code.", "keywords": ["promise", "async", "await", "then"], "hint": "Promises represent future values; async/await makes them easier to read."},
        ],
        "moderate": [
            {"q": "How does event loop tick order affect timers and microtasks?", "keywords": ["event", "loop", "microtask", "macrotask", "queue"], "hint": "Microtasks (promises) run before timers in the same tick."},
            {"q": "What are common causes of React re-render performance issues?", "keywords": ["re-render", "memo", "props", "state", "keys"], "hint": "Unstable props/state and missing memoization trigger extra renders."},
        ],
        "advanced": [
            {"q": "How would you optimize bundle size in a React/Next.js app?", "keywords": ["code split", "tree shaking", "lazy", "cdn", "cache"], "hint": "Code splitting, tree-shaking, lazy loading, CDN + caching."},
        ],
    },
    "python": {
        "basic": [
            {"q": "What are lists vs tuples in Python, and when use each?", "keywords": ["mutable", "immutable", "list", "tuple"], "hint": "Lists are mutable; tuples are immutable and hashable."},
            {"q": "Explain virtual environments and why they matter.", "keywords": ["venv", "virtual", "isolate", "dependency"], "hint": "They isolate dependencies per project."},
        ],
        "moderate": [
            {"q": "How do generators differ from lists in memory usage?", "keywords": ["lazy", "yield", "iterator", "memory"], "hint": "Generators are lazy and keep O(1) memory."},
            {"q": "Describe how GIL affects Python concurrency.", "keywords": ["gil", "thread", "cpu", "io"], "hint": "GIL limits CPU-bound threads; use multiprocessing for CPU, threads/async for IO."},
        ],
        "advanced": [
            {"q": "Outline a production-ready async web stack in Python.", "keywords": ["asyncio", "uvloop", "fastapi", "daphne", "gunicorn"], "hint": "FastAPI/ASGI with uvloop/gunicorn workers; proper timeouts and monitoring."},
        ],
    },
}

DEFAULT_TECH = "general"

class ConversationPhase(str, enum.Enum):
    INIT = "INIT"
    GREETING = "GREETING"
    AI_SPEAKING = "AI_SPEAKING"
    WAIT_FOR_USER = "WAIT_FOR_USER"
    PROCESSING_USER = "PROCESSING_USER"
    QUESTION = "QUESTION"
    EVALUATION = "EVALUATION"
    RETRY = "RETRY"
    EXPLANATION = "EXPLANATION"
    SUMMARY = "SUMMARY"
    END = "END"


class ConversationSession:
    """Holds call-scoped state for the entire conversation lifetime.

    This object is intentionally ephemeral (in-memory) and owned by the
    WebRTCManager. It stores the selected persona, current phase, structured
    history (role-tagged), scores/metrics and any contextual variables the
    controller needs to make deterministic decisions.
    """
    def __init__(self, room_id: str, persona_slug: str, persona_config: Dict[str, Any]):
        self.room_id = room_id
        self.persona_slug = persona_slug
        self.persona_config = persona_config
        self.phase: ConversationPhase = ConversationPhase.INIT
        self.history: List[Dict[str, Any]] = []
        self.score: float = 0.0
        self.metrics: Dict[str, Any] = {}
        self.vars: Dict[str, Any] = {}  # extensible context
        self.started: bool = False       # prevent duplicate greetings
        # Deterministic turn ownership tracking
        self.last_ai_turn_id: Optional[str] = None
        self.last_user_turn_id: Optional[str] = None
        self.expected_user_utterance_id: Optional[str] = None
        self.active_user_utterance_id: Optional[str] = None
        # Track greeting completion to ensure it finishes before user input is accepted
        self.greeting_sent_and_played: bool = False

    def add_user_turn(self, text: str, turn_id: Optional[str] = None, utterance_id: Optional[str] = None):
        entry: Dict[str, Any] = {"role": "user", "content": text}
        if turn_id:
            entry["turn_id"] = turn_id
        if utterance_id:
            entry["utterance_id"] = utterance_id
        self.history.append(entry)

    def add_ai_turn(self, text: str, turn_id: Optional[str] = None):
        entry: Dict[str, Any] = {"role": "assistant", "content": text}
        if turn_id:
            entry["turn_id"] = turn_id
        self.history.append(entry)


class ConversationController:
    """Deterministic controller that owns phase transitions.

    The controller uses the provided `orchestrator` for LLM/STT/TTS operations
    but itself decides phase transitions and when to invoke evaluation.
    
    CRITICAL: All public methods are protected by self._lock to ensure
    serial execution (no concurrent phase transitions or utterance processing).
    """
    def __init__(self, session: ConversationSession, orchestrator, send_audio_callable, notify_callback=None):
        self.session = session
        self.orchestrator = orchestrator
        # send_audio_callable should be an async function that accepts bytes and enqueues/streams them
        self.send_audio = send_audio_callable
        # notify_callback: async callable(dict) -> None for UI events (transcripts)
        self._notify_callback = notify_callback
        # Lock to serialize all public operations
        self._lock = asyncio.Lock()
        # Track if greeting has been sent to prevent double greetings
        self._greeting_sent = False
        # Monotonic counters for deterministic turn/utterance IDs
        self.session.vars.setdefault('ai_turn_seq', 0)
        self.session.vars.setdefault('user_turn_seq', 0)
        self.session.vars.setdefault('utterance_seq', 0)

    async def start(self):
        """Begin the conversation lifecycle: move to GREETING and speak first.
        
        CRITICAL: This is idempotent - can be called multiple times but only
        sends greeting once.
        """
        async with self._lock:
            # Check BOTH session.started AND our own flag for safety
            if self._greeting_sent or self.session.started:
                logger.info("start() called but greeting already sent; skipping duplicate greeting")
                return
            
            logger.info(f"🎙️ Starting conversation: sending greeting")
            
            # Wait for TTS warmup to complete before sending greeting
            # This ensures the first synthesis doesn't block on model load
            await self._wait_for_tts_warmup()
            
            self.session.phase = ConversationPhase.GREETING
            # Craft a succinct greeting using persona identity
            greeting = self._build_greeting()
            ai_turn_id = self._next_ai_turn_id()
            self.session.last_ai_turn_id = ai_turn_id
            self.session.add_ai_turn(greeting, ai_turn_id)
            
            # Notify UI of assistant message before audio, for responsiveness
            await self._notify_chat('assistant', greeting, turn_id=ai_turn_id)

            # Synthesize and stream
            try:
                self.session.phase = ConversationPhase.AI_SPEAKING
                audio = await self.orchestrator.tts.synthesize(greeting, self.session.persona_config.get('voice', {}))
                await self.send_audio(audio, self.session.last_ai_turn_id)
                logger.info(f"✅ Greeting sent and played")
            except Exception as e:
                logger.error(f"❌ Greeting TTS failed: {e}", exc_info=True)
                # If TTS fails, still progress to WAIT_FOR_USER so call doesn't block
            
            await self._set_waiting_for_user()
            self.session.started = True
            self._greeting_sent = True
            # Mark greeting as fully played (allow user input to start now)
            self.session.greeting_sent_and_played = True
            logger.info(f"✅ Greeting lifecycle complete for room {self.session.room_id}")

    def _build_greeting(self) -> str:
        # Prefer persona-configured greeting if provided
        cfg = self.session.persona_config
        configured = cfg.get('greeting')
        if configured and isinstance(configured, str) and configured.strip():
            return configured.strip()
        # Fallback: construct a simple, single-turn greeting
        name = cfg.get('display_name') or self.session.persona_slug
        return f"Hello, I'm {name}. I'm ready when you are."

    async def handle_user_utterance(self, text: str, utterance_id: Optional[str] = None):
        """Main entrypoint for user input. Controller decides how to evaluate and proceed.
        
        CRITICAL: This is protected by self._lock to ensure one utterance is processed
        at a time (no concurrent phase transitions).
        """
        logger.info(f"🎯 handle_user_utterance called with text='{text}' utterance_id='{utterance_id}'")
        logger.info(f"   Current phase: {self.session.phase} | expected={self.session.expected_user_utterance_id} | active={self.session.active_user_utterance_id}")
        
        async with self._lock:
            # GUARD: Reject user input until greeting has fully completed and played
            if not self.session.greeting_sent_and_played:
                logger.warning(f"⚠️ Rejecting user input because greeting has not finished playing yet")
                return
            
            expected = self.session.expected_user_utterance_id

            if self.session.phase != ConversationPhase.WAIT_FOR_USER:
                logger.warning(f"⚠️ Input ignored because phase is {self.session.phase}; only WAIT_FOR_USER accepts utterances")
                return

            if expected and utterance_id and utterance_id != expected:
                logger.warning(f"⚠️ Stale or unexpected utterance_id '{utterance_id}' (expected '{expected}') - discarding")
                return

            if self.session.active_user_utterance_id:
                logger.warning(f"⚠️ Already processing utterance {self.session.active_user_utterance_id}; rejecting new input")
                return

            if not utterance_id:
                utterance_id = expected or self._new_utterance_id()

            # Claim the utterance id immediately to block duplicates
            self.session.active_user_utterance_id = utterance_id

            # Gate low-information inputs to avoid premature phase transitions
            # BUT: If it's a low-confidence marker, speak it to user first
            is_low_confidence_msg = any(
                marker in text.lower() 
                for marker in ["didn't catch that", "could you please repeat", "please repeat"]
            )
            
            if is_low_confidence_msg:
                logger.info("🔊 Low-confidence STT detected; speaking clarification request to user")
                # Speak the clarification without adding to history
                try:
                    audio_bytes = await self.orchestrator.tts.synthesize(
                        "Sorry, I didn't catch that. Could you please repeat?",
                        self.session.persona_config.get('voice', {})
                    )
                    if audio_bytes:
                        await self.send_audio(audio_bytes, self.session.last_ai_turn_id)
                        # Brief pause to let audio play
                        await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"❌ Failed to synthesize low-confidence clarification: {e}")
                
                # Stay in WAIT_FOR_USER and release lock
                self.session.active_user_utterance_id = None
                await self._set_waiting_for_user()
                return
            
            if not self._is_meaningful(text):
                logger.info("🛑 Ignoring low-information/placeholder utterance; staying in WAIT_FOR_USER")
                self.session.active_user_utterance_id = None
                await self._set_waiting_for_user()
                return

            # Add user input to history FIRST before any processing
            user_turn_id = self._next_user_turn_id()
            self.session.last_user_turn_id = user_turn_id
            self.session.add_user_turn(text, user_turn_id, utterance_id)
            logger.info(f"   ✅ Added user turn to history (total turns: {len(self.session.history)})")

            # Now evaluate and generate next question
            flow = self.session.persona_config.get('flow', 'interview')
            logger.info(f"   📋 Flow mode: {flow}")
            
            if flow == 'interview':
                # Deterministic interview flow: question progression + evaluation
                self.session.phase = ConversationPhase.PROCESSING_USER
                logger.info(f"   🔄 Transitioning to PROCESSING_USER (interview flow) phase")
                await self._run_interview_flow(text)
            else:
                # expert/reactive persona: generate an answer and speak
                self.session.phase = ConversationPhase.PROCESSING_USER
                logger.info(f"   🔄 Transitioning to PROCESSING_USER (question) phase, generating answer")
                await self._generate_and_speak_answer(text)

            # Mark utterance as fully processed
            self.session.active_user_utterance_id = None

    async def _evaluate_answer(self, text: str):
        """Generate follow-up question based on user input (interview flow).
        
        In interview mode, we always generate a contextual follow-up question
        based on the user's response for natural conversation flow.
        
        NOTE: The user input has ALREADY been added to history before this
        method is called, so self.session.history is current and complete.
        """
        logger.info(f"🔍 _evaluate_answer: Processing answer: '{text}'")
        try:
            persona_cfg = self.session.persona_config
            logger.info(f"   🤔 Interview mode: generating contextual follow-up question...")
            
            # Generate next question based on COMPLETE conversation history
            # (history already includes the current user input)
            question = await self.orchestrator.generate_question(persona_cfg, self.session.history)
            logger.info(f"   📤 Generated question: '{question}'")
            
            # Add AI response to history BEFORE speaking
            ai_turn_id = self._next_ai_turn_id()
            self.session.last_ai_turn_id = ai_turn_id
            self.session.add_ai_turn(question, ai_turn_id)
            # Notify UI immediately
            await self._notify_chat('assistant', question, turn_id=ai_turn_id)
            
            logger.info(f"   🔊 Synthesizing question to speech...")
            self.session.phase = ConversationPhase.AI_SPEAKING
            audio = await self.orchestrator.tts.synthesize(question, persona_cfg.get('voice', {}))
            logger.info(f"   ✅ TTS complete ({len(audio)} bytes), sending audio...")
            
            # This call blocks until audio finishes playing
            await self.send_audio(audio, self.session.last_ai_turn_id)
            
            # Return to listening state
            await self._set_waiting_for_user()
            logger.info(f"   ✅ Evaluation complete, back to WAIT_FOR_USER")

        except Exception as e:
            logger.error(f"❌ Error in _evaluate_answer: {e}", exc_info=True)
            # Fallback: send a generic response
            try:
                fallback = "I'm listening. Please tell me more."
                self.session.phase = ConversationPhase.AI_SPEAKING
                audio = await self.orchestrator.tts.synthesize(fallback, persona_cfg.get('voice', {}))
                await self.send_audio(audio, self.session.last_ai_turn_id)
            except Exception as e2:
                logger.error(f"❌ Fallback TTS also failed: {e2}")
            # On any error, fallback to safe state
            await self._set_waiting_for_user()

    async def _wait_for_tts_warmup(self):
        """Wait for TTS model pre-warming to complete.
        
        If TTS_PROVIDER is 'coqui', the background warmup thread may still be
        initializing the model. This method waits up to 20 seconds for completion.
        """
        try:
            import os
            provider = os.environ.get("TTS_PROVIDER", "coqui").lower()
            if provider != "coqui":
                # Non-Coqui providers initialize instantly
                return
            
            from ai_personas import tts_providers
            
            # Poll the warmup status with timeout
            max_wait = 20  # seconds
            elapsed = 0
            poll_interval = 0.5
            
            while elapsed < max_wait:
                # Check if warmup is done or if TTS instance is already initialized
                if tts_providers._warmup_done or tts_providers.CoquiTTSProvider._tts_instance:
                    logger.info(f"✅ TTS warmup ready (waited {elapsed:.1f}s)")
                    return
                
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            # Timeout - warmup is still running, but we can't wait forever
            # Proceed anyway; TTS synthesis will just take longer on first call
            logger.warning(f"⚠️  TTS warmup still in progress after {max_wait}s; proceeding (first synthesis may be slow)")
            
        except Exception as e:
            logger.debug(f"TTS warmup check failed: {e}; proceeding anyway")

    async def _generate_and_speak_answer(self, user_text: str):
        logger.info(f"💬 _generate_and_speak_answer called with: '{user_text}'")
        try:
            persona_cfg = self.session.persona_config
            logger.info(f"   Calling orchestrator.generate_answer...")
            answer = await self.orchestrator.generate_answer(persona_cfg, user_text, self.session.history)
            logger.info(f"   📝 LLM generated answer: '{answer}'")
            ai_turn_id = self._next_ai_turn_id()
            self.session.last_ai_turn_id = ai_turn_id
            self.session.add_ai_turn(answer, ai_turn_id)
            await self._notify_chat('assistant', answer, turn_id=ai_turn_id)
            logger.info(f"   🔊 Synthesizing TTS for answer...")
            self.session.phase = ConversationPhase.AI_SPEAKING
            audio = await self.orchestrator.tts.synthesize(answer, persona_cfg.get('voice', {}))
            logger.info(f"   ✅ TTS complete, sending audio ({len(audio)} bytes)...")
            await self.send_audio(audio, self.session.last_ai_turn_id)
            logger.info(f"   ✅ Audio sent, returning to WAIT_FOR_USER")
            await self._set_waiting_for_user()
        except Exception as e:
            logger.error(f"❌ Error in _generate_and_speak_answer: {e}", exc_info=True)
            await self._set_waiting_for_user()

    # -------------------- Interview Flow Helpers --------------------

    def _init_interview_state(self):
        if 'interview' not in self.session.vars:
            self.session.vars['interview'] = {
                "tech": None,
                "tech_confirmed": False,
                "stage": "tech_selection",  # tech_selection -> basic -> moderate -> advanced
                "question_index": 0,
                "current_question": None,
                "current_attempts": 0,
                "asked_in_stage": 0,
                "correct_in_stage": 0,
                "awaiting_answer": False,
                "hint_given": False,  # NEW: Track if hint was just given (wait for retry)
            }
        return self.session.vars['interview']

    def _detect_technology(self, text: str) -> str:
        lowered = (text or "").lower()
        tech_map = {
            "javascript": ["javascript", "js", "react", "next"],
            "python": ["python", "django", "flask", "fastapi"],
            "devops": ["docker", "kubernetes", "k8s", "ci", "cd", "terraform"],
            "system design": ["system", "architecture", "design"],
        }
        for tech, kws in tech_map.items():
            if any(kw in lowered for kw in kws):
                logger.info(f"   📌 Interview detected technology: {tech}")
                return tech
        
        # NEW: If no tech detected and text looks like error/repeat, prompt for explicit choice
        if any(marker in lowered for marker in ["didn't", "error", "repeat", "sorry"]):
            logger.warning(f"   ⚠️  Tech detection got error message or unclear response; need explicit tech")
            return None  # Will trigger explicit tech prompt
        
        logger.info(f"   ℹ️ Tech detection unclear; defaulting to {DEFAULT_TECH}")
        return DEFAULT_TECH

    def _get_question_set(self, tech: str):
        return QUESTION_BANK.get(tech, QUESTION_BANK[DEFAULT_TECH])

    def _pick_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        qs = self._get_question_set(state['tech'])
        stage = state['stage'] if state['stage'] in qs else 'basic'
        questions = qs.get(stage, qs['basic'])
        idx = state['question_index'] % len(questions)
        question = questions[idx]
        state['current_question'] = question
        state['current_attempts'] = 0
        state['awaiting_answer'] = True
        return question

    def _score_answer(self, answer: str, question: Dict[str, Any]) -> Dict[str, Any]:
        ans_lower = (answer or "").lower()
        keywords = question.get('keywords', [])
        hits = sum(1 for kw in keywords if kw in ans_lower)
        needed = max(1, len(keywords) // 2)
        if hits >= needed:
            verdict = "correct"
        elif hits > 0:
            verdict = "partial"
        else:
            verdict = "incorrect"
        return {
            "verdict": verdict,
            "hits": hits,
            "needed": needed,
            "hint": question.get('hint', '')
        }

    def _advance_stage_if_needed(self, state: Dict[str, Any]):
        # Progression: 3 correct in stage -> next stage
        if state['stage'] == 'basic' and state['correct_in_stage'] >= 3:
            state['stage'] = 'moderate'
            state['question_index'] = 0
            state['asked_in_stage'] = 0
            state['correct_in_stage'] = 0
            return "Great, moving to moderate questions."
        if state['stage'] == 'moderate' and state['correct_in_stage'] >= 3:
            state['stage'] = 'advanced'
            state['question_index'] = 0
            state['asked_in_stage'] = 0
            state['correct_in_stage'] = 0
            return "Nice work. Let's tackle some advanced questions."
        return None

    async def _run_interview_flow(self, user_text: str):
        """Deterministic interview flow with staged difficulty and hinting."""
        state = self._init_interview_state()
        persona_cfg = self.session.persona_config

        try:
            # Step 0: explicit technology selection gate
            if state['stage'] == 'tech_selection':
                if not state['awaiting_answer']:
                    # Ask which tech to focus on
                    tech_prompt = (
                        "Which technology should we focus on? Options include JavaScript/React, "
                        "Python/Backend, DevOps, or System Design. If you prefer another, just say your main stack."
                    )
                    state['awaiting_answer'] = True
                    await self._speak_and_wait(tech_prompt, persona_cfg)
                    return
                else:
                    # Use user's reply to set tech, then proceed to basics
                    detected_tech = self._detect_technology(user_text)
                    
                    # NEW: If tech detection unclear (returned None), re-prompt
                    if detected_tech is None:
                        retry_prompt = (
                            "I didn't catch that. Can you say your primary tech? "
                            "Python, JavaScript, DevOps, or something else?"
                        )
                        state['awaiting_answer'] = True
                        await self._speak_and_wait(retry_prompt, persona_cfg)
                        return
                    
                    state['tech'] = detected_tech
                    state['tech_confirmed'] = True
                    state['stage'] = 'basic'
                    state['awaiting_answer'] = False
                    opener = f"Great, we'll focus on {state['tech']}. Let's start with fundamentals."
                    question = self._pick_question(state)
                    response = f"{opener} {question['q']}"
            
            # NEW: Handle hint retry - user already received hint, now evaluating second attempt
            elif state['hint_given']:
                # User is responding to hint, evaluate again
                result = self._score_answer(user_text, state['current_question'])
                state['asked_in_stage'] += 1
                state['hint_given'] = False
                
                feedback_parts = []
                if result['verdict'] == 'correct' or result['verdict'] == 'partial':
                    # Accept partial/correct on second attempt
                    state['correct_in_stage'] += 1
                    feedback_parts.append("Great!")
                else:
                    # Still wrong after hint; move on with core concept
                    feedback_parts.append("Let me share the key idea:")
                    feedback_parts.append(result['hint'])

                # Move to next question
                state['question_index'] += 1
                stage_transition = self._advance_stage_if_needed(state)
                question = self._pick_question(state)
                
                response_parts = feedback_parts
                if stage_transition:
                    response_parts.append(stage_transition)
                response_parts.append(f"Next question: {question['q']}")
                response = " ".join(p for p in response_parts if p)
            
            # Evaluate user's answer to current question
            elif state['awaiting_answer'] and state['current_question']:
                result = self._score_answer(user_text, state['current_question'])
                
                if result['verdict'] == 'correct':
                    # Correct answer - move to next question
                    state['correct_in_stage'] += 1
                    state['asked_in_stage'] += 1
                    feedback_parts = ["Correct!"]
                    
                    # Check for stage transition
                    stage_transition = self._advance_stage_if_needed(state)
                    state['question_index'] += 1
                    question = self._pick_question(state)
                    
                    response_parts = feedback_parts
                    if stage_transition:
                        response_parts.append(stage_transition)
                    response_parts.append(f"Next question: {question['q']}")
                    response = " ".join(p for p in response_parts if p)
                    
                elif result['verdict'] == 'partial' and state['current_attempts'] == 0:
                    # First incorrect/partial attempt - give hint and wait for retry
                    state['current_attempts'] += 1
                    state['hint_given'] = True
                    state['awaiting_answer'] = True  # Still waiting, but for retry
                    
                    hint_response = f"I'll share a quick hint: {result['hint']} Please try again."
                    await self._speak_and_wait(hint_response, persona_cfg)
                    return
                    
                else:
                    # Wrong or second incorrect attempt - share concept and move on
                    state['asked_in_stage'] += 1
                    feedback_parts = ["Let me share the key idea:", result['hint']]
                    
                    state['question_index'] += 1
                    stage_transition = self._advance_stage_if_needed(state)
                    question = self._pick_question(state)
                    
                    response_parts = feedback_parts
                    if stage_transition:
                        response_parts.append(stage_transition)
                    response_parts.append(f"Next question: {question['q']}")
                    response = " ".join(p for p in response_parts if p)
            else:
                # No pending question (e.g., first turn after greeting) -> ask next
                question = self._pick_question(state)
                response = f"Let's continue. {question['q']}"

            # Add to history and speak
            ai_turn_id = self._next_ai_turn_id()
            self.session.last_ai_turn_id = ai_turn_id
            self.session.add_ai_turn(response, ai_turn_id)

            # Notify UI before audio for responsiveness
            await self._notify_chat('assistant', response, turn_id=self.session.last_ai_turn_id)
            logger.info("   🔊 Synthesizing interview question/feedback...")
            self.session.phase = ConversationPhase.AI_SPEAKING
            audio = await self.orchestrator.tts.synthesize(response, persona_cfg.get('voice', {}))
            await self.send_audio(audio, self.session.last_ai_turn_id)
            await self._set_waiting_for_user()
        except Exception as e:
            logger.error(f"❌ Error in interview flow: {e}", exc_info=True)
            await self._set_waiting_for_user()

    async def _speak_and_wait(self, text: str, persona_cfg: Dict[str, Any]):
        """Utility to speak a prompt and return to WAIT_FOR_USER."""
        ai_turn_id = self._next_ai_turn_id()
        self.session.last_ai_turn_id = ai_turn_id
        self.session.add_ai_turn(text, ai_turn_id)
        await self._notify_chat('assistant', text, turn_id=ai_turn_id)
        self.session.phase = ConversationPhase.AI_SPEAKING
        audio = await self.orchestrator.tts.synthesize(text, persona_cfg.get('voice', {}))
        await self.send_audio(audio, self.session.last_ai_turn_id)
        await self._set_waiting_for_user()

    def _next_ai_turn_id(self) -> str:
        self.session.vars['ai_turn_seq'] += 1
        return f"ai-{self.session.vars['ai_turn_seq']}"

    def _next_user_turn_id(self) -> str:
        self.session.vars['user_turn_seq'] += 1
        return f"user-{self.session.vars['user_turn_seq']}"

    def _new_utterance_id(self) -> str:
        self.session.vars['utterance_seq'] += 1
        return f"utt-{self.session.vars['utterance_seq']}"

    async def _set_waiting_for_user(self):
        """Reset to WAIT_FOR_USER, clear active flags, and prime the next expected utterance."""
        self.session.phase = ConversationPhase.WAIT_FOR_USER
        self.session.active_user_utterance_id = None
        self.session.expected_user_utterance_id = self._new_utterance_id()
        logger.info(f"🎧 Waiting for user; expecting utterance_id={self.session.expected_user_utterance_id}")

    def _is_meaningful(self, text: str) -> bool:
        """Heuristic to filter placeholder or trivial utterances.
        - Reject empty/very short inputs
        - Reject known mock placeholders
        - Reject repeated duplicates of last user turn (same person talking twice)
        - Reject plain greetings when in WAIT_FOR_USER immediately after greeting
        """
        if not text:
            return False
        
        s = text.strip()
        
        # Too short to be meaningful
        if len(s) < 2:
            return False
        
        # Known mock STT placeholders - should never be heard
        placeholders = {
            "hello there, this is a mock transcription.",
            "mock transcription",
        }
        if s.lower() in placeholders:
            logger.info(f"   ⏭️  Skipping known mock placeholder: '{s}'")
            return False
        
        # Trivial single-word greetings
        trivial = {"hi", "hello", "hey", "test", "...", "ok", "okay", "yes", "no"}
        if s.lower() in trivial:
            # Allow single words in initial greeting phase
            if self.session.phase != ConversationPhase.GREETING:
                logger.info(f"   ⏭️  Skipping trivial single-word utterance: '{s}'")
                return False

        # Low-confidence STT fallback phrases should not advance the flow
        low_confidence_markers = {
            "i didn't catch that, could you please repeat?",
            "could you please repeat",
            "please repeat that",
        }
        if any(marker in s.lower() for marker in low_confidence_markers):
            logger.info("   ⏭️  Skipping low-confidence STT fallback; asking user to repeat")
            return False
        
        # Check for duplicate of last user turn (prevents processing same input twice)
        for item in reversed(self.session.history):
            if item.get('role') == 'user':
                last_user = item.get('content', '').strip().lower()
                if last_user and last_user == s.lower():
                    logger.info(f"   ⏭️  Skipping duplicate user turn: '{s}'")
                    return False
                # Only check the last user turn, then break
                break
        
        return True

    async def _notify_chat(self, role: str, text: str, turn_id: Optional[str] = None, utterance_id: Optional[str] = None):
        """Emit a chat/transcript event to UI if a notifier is configured."""
        try:
            if not self._notify_callback or not text:
                return
            message = {
                'type': 'transcript',
                'role': role,
                'text': text,
                'turnId': turn_id,
                'utteranceId': utterance_id,
                'roomId': self.session.room_id,
                'timestamp': time.time(),
            }
            await self._notify_callback(message)
        except Exception as e:
            logger.debug(f"notify_chat failed: {e}")
