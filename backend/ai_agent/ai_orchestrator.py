import logging
import asyncio
import time
import re
from collections import Counter
from typing import Optional
from django.conf import settings
# from ai_personas.models import Persona # Avoid circular import if possible, but used inside method
from .providers import get_providers
from .utils import build_prompt
from .tts_stream_track import TTSStreamTrack
from ai_personas.tts_providers import ProviderFactory

logger = logging.getLogger(__name__)

class AIOrchestrator:
    def __init__(self):
        # We rely on the factory now
        # However, to keep existing signature if any, we just init providers lazily
        self._providers = None
        
        # History storage
        self.call_history = {} 
        # Structured per-room memory
        self.memory_by_room: dict[str, dict] = {}

    @property
    def providers(self):
        if not self._providers:
            # Re-fetch providers from ai_agent.providers (correct module with STT_PROVIDER support)
            self._providers = get_providers()
        return self._providers

    @property
    def stt(self):
        return self.providers['stt']

    @property
    def llm(self):
        return self.providers['llm']

    @property
    def tts(self):
        return self.providers['tts']

    async def handle_utterance(self, room_id, audio_bytes, persona_slug, peer_connection=None):
        """
        Process a complete user utterance: VAD -> STT -> Prompt -> LLM -> TTS -> Streaming
        peer_connection: The aiortc RTCPeerConnection object, needed for streaming audio back.
        """
        start_time = time.time()
        logger.info(f"Orchestrator handling utterance for room {room_id}, persona {persona_slug}")

        # 0. Get Persona Config
        from ai_personas.models import Persona
        from asgiref.sync import sync_to_async
        
        try:
            persona = await sync_to_async(Persona.objects.get)(slug=persona_slug)
            persona_config = persona.config
        except Persona.DoesNotExist:
            logger.error(f"Persona {persona_slug} not found")
            return None

        # 1. STT
        stt_start = time.time()
        transcript = await self.stt.transcribe(audio_bytes)
        stt_end = time.time()
        
        if not transcript.strip():
            return None
            
        logger.info(f"STT: '{transcript}'")

        # 2. History
        if room_id not in self.call_history:
            self.call_history[room_id] = []
        history = self.call_history[room_id]
        memory = self.memory_by_room.setdefault(room_id, {
            'current_topic': None,
            'last_intent': None,
            'open_questions': [],
            'confirmed_facts': [],
        })

        # 3. Build Prompt
        # Include memory context as an extra system message to ground the LLM
        messages = build_prompt(persona_config, history, transcript)
        mem_context = []
        if memory.get('current_topic'):
            mem_context.append(f"current_topic={memory['current_topic']}")
        if memory.get('confirmed_facts'):
            facts = "; ".join(map(str, memory['confirmed_facts'][:5]))
            mem_context.append(f"facts={facts}")
        if mem_context:
            messages.insert(0, {'role': 'system', 'content': f"Conversation memory: {', '.join(mem_context)}"})

        # 4. LLM
        llm_start = time.time()
        # Moderation check (stub)
        if persona_config.get("moderation", {}).get("profanity_filter", True):
            pass

        try:
            response_data = await self.llm.generate_response(messages, persona_config.get("system_prompt", ""))
            if isinstance(response_data, str):
                import json
                try:
                   response_data = json.loads(response_data)
                except:
                   response_data = {"text": response_data, "should_tts": True}
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            response_data = {"text": "I encountered an error.", "should_tts": True}

        llm_end = time.time()
        llm_text = response_data.get("text", "")
        should_tts = response_data.get("should_tts", True)
        logger.info(f"LLM: '{llm_text}'")

        # Update History
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": llm_text})
        if len(history) > 20: 
            self.call_history[room_id] = history[-20:]

        # Simple memory update: set topic if detectable from last turn
        try:
            from .intent import resolve_intent
            intent = resolve_intent(transcript, memory)
            memory['last_intent'] = intent.get('intent_type')
            if intent.get('topic'):
                memory['current_topic'] = intent['topic']
        except Exception:
            pass

        tts_audio = None
        tts_start = 0
        tts_end = 0
        tts_bytes_len = 0
        tts_provider_name = "Unknown"

        if should_tts and llm_text:
            # 5. TTS / Streaming
            tts_start = time.time()
            voice_config = persona_config.get("voice", {})
            
            # Decide if we stream or sync
            # New flow: Always try to stream if peer_connection is available
            # But the provider might only support sync internally (Prebuffer)
            # The interface is stream() -> AsyncIterator
            
            # Using ProviderFactory specifically to ensure we can swap if needed per voice config
            # But currently we use the global configured provider
            # Ideally voice_config should specify provider, but we simplified to environment/global for MVP
            
            try:
                # We can launch streaming as a background task if we had a proper
                # mechanism to attach track dynamically without blocking wait.
                # aiortc allows adding tracks dynamically but it's complex on the fly.
                # EASIER MVP: Wait for synthesis stream to be ready (chunk iterator) 
                # and attach track immediately.
                
                # Check if we have a peer connection to stream to
                if peer_connection:
                    logger.info("Starting TTS Stream...")
                    
                    # Call stream() to get iterator
                    # Note: For Coqui provider, this will internally wait for synthesis (Prebuffer)
                    # then yield chunks. So this await might block for synthesis duration unless provider is truly streaming.
                    # Our Coqui implementation is currently Prebuffer (blocks for synthesis, then yields).
                    # So latency here is Synthesis Time.
                    
                    chunk_iterator = self.tts.stream(llm_text, voice_config)
                    
                    # Create Track
                    track = TTSStreamTrack(chunk_iterator)
                    
                    # Attach to PC
                    # We need to find the transceiver or add data.
                    # Typically we add track to the PC.
                    sender = peer_connection.addTrack(track)
                    
                    # We need to cleanup track after it ends? 
                    # The track emits 'ended' event?
                    
                    # For telemetry, we don't know exact end time yet since it plays async.
                    # We'll just mark TTS initialized time.
                    tts_end = time.time()
                    tts_provider_name = type(self.tts).__name__
                    
                else:
                    # Fallback to non-streaming bytes return (legacy behavior)
                    tts_audio = await self.tts.synthesize(llm_text, voice_config)
                    tts_end = time.time()
                    tts_bytes_len = len(tts_audio)
                    tts_provider_name = type(self.tts).__name__
                    logger.info(f"TTS generated {tts_bytes_len} bytes (Non-streaming)")
            
            except Exception as e:
                logger.error(f"TTS Generation Failed: {e}")
                # Fallback?

        total_duration = time.time() - start_time
        
        # Telemetry Log
        metrics = {
            "call_id": room_id,
            "persona": persona_slug,
            "turn_timestamp": start_time,
            "latency_stt": stt_end - stt_start,
            "latency_llm": llm_end - llm_start,
            "latency_tts": (tts_end - tts_start) if tts_end else 0,
            "total_duration": total_duration,
            "tts_provider": tts_provider_name,
            "tts_bytes": tts_bytes_len
        }
        print(f"TELEMETRY: {metrics}") 

        result = {
            "transcript": transcript,
            "response_text": llm_text,
            "tts_audio": tts_audio,
            "metrics": metrics
        }
        return result

    # --- New helper LLM modes for deterministic controller usage ---
    async def generate_question(self, persona_config, history):
        """Generate a short, voice-friendly question based on persona flow and history."""
        # Build concise prompt in text mode (no JSON instruction)
        # System prompt is already included in messages by build_prompt
        prompt = build_prompt(persona_config, history, None, mode="text")
        try:
            # Pass empty system_prompt since it's already in messages
            out = await self.llm.generate_response(prompt, "")
            if isinstance(out, str):
                result = out[:500]  # Safeguard: questions should be short
            else:
                result = out.get('text', '')[:500]
            return result if result.strip() else "Can you tell me more about that?"
        except Exception as e:
            logger.error(f"generate_question error: {e}")
            return "Can you tell me more about that?"

    async def generate_answer(self, persona_config, user_text, history):
        logger.info(f"🤖 Orchestrator.generate_answer: user_text='{user_text[:100]}'")
        # Ask LLM to answer user_text in persona tone (text mode, no JSON)
        # System prompt already included in messages
        prompt = build_prompt(persona_config, history, user_text, mode="text")
        # Inject memory context if available (history may not include it yet)
        try:
            # history items are not tied to a specific room here; safe to include generic memory marker
            # Skip if no memory set in orchestrator for this flow
            mem_ctx = getattr(self, 'memory_by_room', None)
            if mem_ctx:
                # In this text-only method we cannot know room_id; include a neutral instruction
                prompt.insert(0, {'role': 'system', 'content': 'If the system provides context memory, align answers to current topic and confirmed facts.'})
        except Exception:
            pass
        logger.info(f"   Built prompt with {len(prompt)} messages")
        try:
            logger.info(f"   Calling LLM.generate_response...")
            # Pass empty system_prompt since it's already in messages
            out = await self.llm.generate_response(prompt, "")
            logger.info(f"   LLM returned: {str(out)[:200]}...")
            if isinstance(out, str):
                response_text = out
            else:
                response_text = out.get('text', '')

            # Smart truncation: only truncate at sentence boundaries if > 1800 chars
            # This allows full, coherent answers without mid-sentence cuts
            response_text = (response_text or "").strip()
            max_response_length = 1800  # Much longer than 600 to allow detailed answers
            
            if len(response_text) > max_response_length:
                # Truncate at sentence boundary (period, question mark, exclamation)
                truncated = response_text[:max_response_length]
                
                # Find last sentence boundary
                last_period = max(
                    truncated.rfind('.'),
                    truncated.rfind('?'),
                    truncated.rfind('!')
                )
                
                if last_period > max_response_length * 0.7:  # Only truncate if found in last 30%
                    response_text = truncated[:last_period + 1]
                    logger.info(f"✂️ Truncated response at sentence boundary ({len(response_text)} chars from {len(out) if isinstance(out, str) else len(out.get('text', ''))} original)")
                else:
                    # If no clear boundary, just truncate and append ellipsis
                    response_text = truncated.rstrip() + "..."
                    logger.info(f"✂️ Truncated response (no sentence boundary found)")

            # Basic gibberish detection: repeated tokens or very low vocabulary variety
            if response_text:
                tokens = [t for t in re.split(r"\s+", response_text.lower()) if t]
                if tokens:
                    token_counts = Counter(tokens)
                    max_freq = max(token_counts.values()) / len(tokens)
                    unique_ratio = len(token_counts) / len(tokens)
                    # Look for repeated trigrams
                    trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
                    trigram_counts = Counter(trigrams) if trigrams else {}
                    trigram_max = (max(trigram_counts.values()) / len(trigrams)) if trigrams else 0.0

                    if (len(tokens) >= 20 and (max_freq > 0.35 or unique_ratio < 0.30 or trigram_max > 0.20)):
                        logger.warning("⚠️  Detected low-quality/gibberish LLM response; substituting safe reply")
                        response_text = "I didn't quite follow that. Could you restate what you're working on or the problem you're trying to solve?"

            if response_text:
                logger.info(f"   ✅ Returning text: '{response_text[:100]}'")
                return response_text
            return "I'm sorry, I couldn't generate an answer right now."
        except Exception as e:
            logger.error(f"❌ generate_answer error: {e}", exc_info=True)
            return "I'm sorry, I couldn't generate an answer right now."

    async def evaluate_answer(self, persona_config, history, user_text):
        """Ask the LLM to evaluate `user_text` against expected criteria and return structured JSON.

        Expected result example: {"decision": "advance|retry|explain|continue", "score": 1.0, "explanation":"..."}
        """
        # Build prompt in JSON mode with evaluation instruction
        # System prompt already included by build_prompt
        messages = build_prompt(persona_config, history, user_text, mode="json")
        # Append evaluation-specific instruction
        eval_prompt = {
            'role': 'system',
            'content': (
                'You are an impartial evaluator. Given the persona goals and the user answer, '
                'assess correctness and return a JSON object with keys: decision, score, explanation. '
                'Allowed decisions: advance, retry, explain, continue. Score range 0.0-1.0.'
            )
        }
        messages.append(eval_prompt)

        try:
            # Pass empty system_prompt since it's already in messages
            raw = await self.llm.generate_response(messages, "")
            # Expect JSON string or dict
            if isinstance(raw, dict):
                return raw
            import json
            try:
                parsed = json.loads(str(raw)[:2000])  # Safeguard: limit parse size
                return parsed
            except Exception:
                # Fallback: return conservative continue
                return {"decision": "continue", "score": 0.0, "explanation": str(raw)[:500]}
        except Exception as e:
            logger.error(f"evaluate_answer error: {e}")
            return {"decision": "continue", "score": 0.0, "explanation": "Evaluation failed."}
