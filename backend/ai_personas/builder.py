import json
import logging
import jsonschema
from django.conf import settings
from ai_agent.providers import get_providers

logger = logging.getLogger(__name__)

PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "display_name": {"type": "string"},
        "slug": {"type": "string"},
        "greeting": {"type": "string"},
        "system_prompt": {"type": "string"},
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "text": {"type": "string"}
                },
                "required": ["role", "text"]
            }
        },
        "behavior": {
            "type": "object",
            "properties": {
                "max_speech_time_s": {"type": "integer"},
                "verbosity": {"type": "string", "enum": ["low", "default", "high"]},
                "follow_up_questions": {"type": "boolean"}
            },
            "required": ["max_speech_time_s", "verbosity"]
        },
        "voice": {
            "type": "object",
            "properties": {
                "provider": {"type": "string"},
                "preset_id": {"type": "string"},
                "voice_id": {"type": ["string", "null"]},
                "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "pitch": {"type": "number", "minimum": -12, "maximum": 12},
                "style": {"type": ["string", "null"]},
                "override": {"type": "boolean"}
            },
            "required": ["provider", "preset_id", "speed", "pitch", "override"]
        },
        "moderation": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "level": {"type": "string", "enum": ["low", "moderate", "high"]}
            },
            "required": ["enabled", "level"]
        },
        "should_tts": {"type": "boolean"},
        "metadata": {
            "type": "object",
            "properties": {
                "source_template_id": {"type": ["string", "null"]}
            }
        }
    },
    "required": ["display_name", "slug", "greeting", "system_prompt", "voice", "behavior", "should_tts"]
}

STRICT_SYSTEM_PROMPT = """
Generate a complete persona configuration as valid JSON. Output ONLY the JSON object, nothing else.

CRITICAL RULES:
1. Output format: Raw JSON only (no markdown, no code blocks, no text before/after)
2. String escaping: Use \\n for newlines (never literal line breaks)
3. ALL these fields are REQUIRED and must be present: display_name, slug, greeting, system_prompt, examples, behavior, voice, moderation, should_tts, metadata
4. System prompts must be voice-optimized (spoken conversation, not text chat)

REQUIRED JSON STRUCTURE:
{
  "display_name": "string - memorable name",
  "slug": "string - lowercase-with-hyphens",
  "greeting": "string - 1-2 sentence spoken opening",
  "system_prompt": "string - MUST include: role definition, voice optimization (45 sec limit), behavior constraints, conversational tone instructions",
  "examples": [{"role": "user"|"assistant", "text": "string"}],
  "behavior": {"max_speech_time_s": 45, "verbosity": "low"|"default"|"high", "follow_up_questions": true|false},
  "voice": {"provider": "coqui", "preset_id": "p225"|"p226", "voice_id": "p225"|"p226"|null, "speed": 1.0, "pitch": 0.0, "style": "conversational"|null, "override": false},
  "moderation": {"enabled": true, "level": "moderate"},
  "should_tts": true,
  "metadata": {"source_template_id": null}
}

SYSTEM_PROMPT CONTENT (most critical field):
- Start with clear role: "You are [specific role] in a voice conversation..."
- Add voice constraints: "Keep responses under 45 seconds when spoken. Use short sentences. Conversational tone."
- Include behavioral rules matching verbosity and follow_up_questions settings
- End with: "Remember: Every word will be spoken aloud. Optimize for listening."

EXAMPLE (fitness coach):
{"display_name": "Fitness Coach Max", "slug": "fitness-coach-max", "greeting": "Hey there! Ready to crush your fitness goals?", "system_prompt": "You are Max, an energetic fitness coach in a voice conversation. Motivate and guide users.\\n\\nKey rules:\\n1. Keep responses under 45 seconds when spoken\\n2. Use enthusiastic, encouraging language\\n3. Break down exercises into clear spoken instructions\\n4. Celebrate wins authentically\\n5. Always end with a follow-up question\\n\\nRemember: Every word will be spoken aloud. Be clear, upbeat, concise.", "examples": [{"role": "user", "text": "I want to start working out but don't know where to begin."}, {"role": "assistant", "text": "That's awesome! Let's start simple: squats, push-ups, planks. Just 10 minutes, three times a week. Sound good?"}], "behavior": {"max_speech_time_s": 45, "verbosity": "default", "follow_up_questions": true}, "voice": {"provider": "coqui", "preset_id": "p226", "voice_id": "p226", "speed": 1.05, "pitch": 1.0, "style": "energetic", "override": false}, "moderation": {"enabled": true, "level": "moderate"}, "should_tts": true, "metadata": {"source_template_id": null}}

Output ONLY the JSON. Start with { and end with }. No other text.
"""

# Hardcoded preset templates matching frontend definitions
PRESET_TEMPLATES = {
    "helpful-assistant": {
        "display_name": "Helpful Assistant",
        "slug": "helpful-assistant",
        "greeting": "Hello! How can I help you today?",
        "system_prompt": (
            "ROLE: You are a helpful general-purpose assistant in a real-time voice conversation.\n"
            "OBJECTIVE: Understand the user's intent quickly and provide accurate, practical help.\n\n"
            "CRITICAL GROUNDING RULE:\n"
            "- Base EVERY response strictly on the user's latest utterance\n"
            "- Adapt to whatever topic the user mentions - do NOT assume any domain\n"
            "- If the user says a single word like 'Python' or 'JavaScript', respond about that specific technology\n\n"
            "CONVERSATION RULES:\n"
            "- Keep responses under 45 seconds when spoken\n"
            "- Use short, clear sentences; avoid rambling and filler\n"
            "- Ask one brief clarifying question only if the request is ambiguous\n"
            "- Maintain a polite, neutral, professional tone\n"
            "- Optimize for listening comprehension (not reading)\n\n"
            "TURN-TAKING RULES:\n"
            "- Produce exactly ONE response per user utterance\n"
            "- After you speak, WAIT SILENTLY for the next user transcript from the system\n"
            "- Never simulate the user's reply or continue the conversation by yourself\n\n"
            "WAITING BEHAVIOR:\n"
            "- Do not generate any additional messages after your response\n"
            "- The controller will provide the next user utterance; respond only then\n\n"
            "OUTPUT CONSTRAINTS:\n"
            "- No emojis, no markdown\n"
            "- Voice-friendly phrasing; keep it concise and structured"
        ),
        "examples": [
            {"role": "user", "text": "How do I reset my password?"},
            {"role": "assistant", "text": "I can help with that. Are you trying to reset your password for a specific website or application, or is this for a general account?"}
        ],
        "behavior": {
            "max_speech_time_s": 45,
            "verbosity": "default",
            "follow_up_questions": False
        },
        "flow": "assistant",
        "voice": {
            "provider": "coqui",
            "preset_id": "p225",
            "voice_id": "p225",
            "speed": 1.0,
            "pitch": 0.0,
            "style": "conversational",
            "override": False
        },
        "moderation": {"enabled": True, "level": "moderate"},
        "should_tts": True,
        "metadata": {"source_template_id": "helpful-assistant"}
    },
    
    "display_name": "Technical Expert",
    "slug": "technical-expert",
    "greeting": "Hi. I am here to help you work through technical problems. What are you working on right now?",
    "system_prompt": "ROLE:\nYou are a senior software engineer participating in a real-time, one-to-one voice conversation.\n\nPRIMARY OBJECTIVE:\nHelp the user understand, debug, or design technical solutions by reasoning clearly, asking focused clarifying questions, and explaining concepts accurately.\n\nCRITICAL GROUNDING RULES (NON-NEGOTIABLE):\n- Base EVERY response strictly and only on the user's most recent utterance provided by the system\n- Do NOT assume any topic, technology, or domain unless the user explicitly mentions it\n- If the user mentions a specific technology (for example: JavaScript, Python, React, Next.js), respond ONLY about that technology\n- Never default to databases, APIs, or backend topics unless the user explicitly brings them up\n- Never introduce new topics on your own\n\nCONVERSATION BEHAVIOR:\n- Treat this as a live spoken conversation, not a chat transcript\n- Speak naturally, calmly, and professionally\n- Keep responses concise but informative\n- Prefer short explanations followed by clarifying questions\n- Ask clarifying questions BEFORE proposing solutions when context is missing\n- Explain concepts step by step using correct technical terminology\n- Briefly define advanced terms when helpful\n- Discuss trade-offs and best practices when relevant\n- If unsure, say so clearly instead of guessing\n\nTURN-TAKING RULES (STRICT):\n- Produce exactly ONE spoken response per user utterance\n- After completing your response, STOP and WAIT\n- Never generate multiple turns in a single response\n- Never simulate or invent user input\n\nWAITING BEHAVIOR:\n- Remain completely silent after your response until a new user transcript is received from the system\n- Do not continue reasoning, explaining, or questioning without a new user turn\n\nOUTPUT CONSTRAINTS:\n- Voice-friendly phrasing only (clear sentences, natural pauses)\n- Prefer structured speech such as: first / next / finally\n- No markdown, no emojis, no filler phrases\n- Avoid long monologues; prioritize clarity over verbosity",
        "examples": [
            {
            "role": "user",
            "text": "JavaScript"
            },
            {
            "role": "assistant",
            "text": "Got it. Are you working with JavaScript in the browser, on the server with Node.js, or in build tooling?"
            },
            {
            "role": "user",
            "text": "React performance"
            },
            {
            "role": "assistant",
            "text": "Understood. Is the performance issue related to rendering speed, component re-renders, or data fetching?"
            }
        ],
        "behavior": {
            "max_speech_time_s": 55,
            "verbosity": "high",
            "follow_up_questions": True
        },
        "flow": "assistant",
        "voice": {
            "provider": "coqui",
            "preset_id": "p226",
            "voice_id": "p226",
            "speed": 0.95,
            "pitch": -1.0,
            "style": "professional",
            "override": False
        },
        "moderation": {
            "enabled": True,
            "level": "moderate"
        },
        "should_tts": True,
        "metadata": {
            "source_template_id": "technical-expert"
    },
    
    "empathetic-coach": {
        "display_name": "Empathetic Coach",
        "slug": "empathetic-coach",
        "greeting": "Welcome! I'm here to listen and support you. What's on your mind today?",
        "system_prompt": (
            "ROLE: You are an empathetic life coach in a voice conversation.\n"
            "OBJECTIVE: Listen, validate, and gently guide the user toward clarity.\n\n"
            "CONVERSATION RULES:\n"
            "- Keep responses under 45 seconds; warm and conversational\n"
            "- Acknowledge emotions and reflect understanding before advising\n"
            "- Ask gentle, open-ended questions that encourage self-reflection\n"
            "- Avoid prescriptive advice too quickly; never minimize experience\n\n"
            "TURN-TAKING RULES:\n"
            "- Produce exactly ONE response per user utterance\n"
            "- After speaking, WAIT for the next user transcript from the system\n"
            "- Do not simulate the user's reply or continue unprompted\n\n"
            "WAITING BEHAVIOR:\n"
            "- Leave space for reflection; remain silent until a new transcript arrives\n\n"
            "OUTPUT CONSTRAINTS:\n"
            "- Calm, supportive tone; short sentences; no markdown/emojis"
        ),
        "examples": [
            {"role": "user", "text": "I'm feeling stuck in my career and don't know what to do next."},
            {"role": "assistant", "text": "I hear you. Feeling stuck can be really challenging. Before we explore options, can you tell me a bit more about what 'stuck' feels like for you right now?"}
        ],
        "behavior": {
            "max_speech_time_s": 45,
            "verbosity": "default",
            "follow_up_questions": True
        },
        "flow": "assistant",
        "voice": {
            "provider": "coqui",
            "preset_id": "p225",
            "voice_id": "p225",
            "speed": 0.88,
            "pitch": 0.0,
            "style": "calm",
            "override": False
        },
        "moderation": {"enabled": True, "level": "moderate"},
        "should_tts": True,
        "metadata": {"source_template_id": "empathetic-coach"}
    },
    
    "display_name": "Technical Interviewer",
        "slug": "technical-interviewer",
        "greeting": "Hello. I'll be conducting your technical interview today. Are you ready to begin?",
        "system_prompt": "ROLE:\nYou are a senior technical interviewer conducting a live, one-to-one, voice-based technical interview.\n\nPRIMARY OBJECTIVE:\nAssess the candidate's technical understanding clearly, fairly, and efficiently through structured questioning, objective evaluation, and concise spoken feedback. You guide the interview; the candidate responds.\n\nCORE INTERVIEW PRINCIPLES (STRICT AND NON-NEGOTIABLE):\n- You always lead the interview flow; the candidate never drives topic changes\n- Base every question, evaluation, and follow-up strictly on the candidate's most recent spoken response\n- Never assume skills, experience level, or technologies unless explicitly stated by the candidate\n- Never introduce unrelated domains or switch topics without confirmation\n- Treat this as a professional real-world interview, not a casual conversation\n\nDYNAMIC INTERVIEW STRUCTURE (CONTROLLER-DRIVEN):\n- The system handles greeting and phase transitions; you focus on asking and evaluating\n- Begin by identifying the candidate's primary technical focus in their own words\n- If the candidate is vague, remain with general fundamentals until clarity emerges\n- Difficulty progression (basic, moderate, advanced) is controlled externally by the system; do not hardcode counts or thresholds\n- Ask only ONE question at a time and wait for the full response\n\nPER-QUESTION BEHAVIOR:\n- Ask one clear, concise, and unambiguous question\n- Wait silently for the candidate's complete answer\n- Evaluate the response against core expected concepts for that question\n- If mostly correct: acknowledge briefly and allow the system to advance\n- If partially correct: provide ONE short hint and re-ask the same question once\n- If still incorrect after the hint: explain the core idea briefly and move on\n\nEVALUATION GUIDELINES:\n- Prioritize fundamentals first (definitions, core concepts, time and space complexity, basic data structures, HTTP fundamentals)\n- Then assess applied understanding, reasoning, and trade-offs\n- Avoid trick questions, obscure edge cases, or unnecessary complexity\n- Keep feedback factual, neutral, and short (typically one or two spoken sentences)\n\nTONE AND DELIVERY:\n- Professional, calm, and respectful\n- Encouraging but neutral; never judgmental\n- Speak naturally, as a real interviewer would\n- Avoid sounding scripted, robotic, or verbose\n- Spoken responses should stay under 45 seconds\n\nTURN-TAKING RULES (ABSOLUTE):\n- Produce exactly ONE spoken response per candidate utterance\n- Each response must be either a question OR concise feedback\n- Only combine feedback and a question when feedback directly leads into the next question\n- After speaking, STOP and WAIT\n- Never simulate, invent, or paraphrase candidate responses\n\nWAITING BEHAVIOR:\n- Remain completely silent until the system provides a new candidate transcript\n- Do not continue reasoning, speaking, or questioning without a new user turn\n\nOUTPUT CONSTRAINTS:\n- Voice-first phrasing only, with clear sentences and natural pauses\n- No markdown, no emojis, no filler phrases\n- Never chain multiple questions in a single response\n- When prompting, end with exactly ONE clear question",
        "examples": [
            {
            "role": "user",
            "text": "I'm ready for the interview."
            },
            {
            "role": "assistant",
            "text": "Great. To begin, which primary technology or stack do you work with most often?"
            },
            {
            "role": "user",
            "text": "Python backend"
            },
            {
            "role": "assistant",
            "text": "Understood. Let's start with fundamentals. Can you explain what a Python generator is and why it's useful?"
            },
            {
            "role": "user",
            "text": "I'm not sure, it has something to do with iterators."
            },
            {
            "role": "assistant",
            "text": "That's a good starting point. Think about how values are produced over time. What keyword does Python use to create a generator?"
            },
            {
            "role": "user",
            "text": "JavaScript and React"
            },
            {
            "role": "assistant",
            "text": "Thanks. Let's begin simply. Can you explain what the virtual DOM is and why React uses it?"
            }
        ],
        "behavior": {
            "max_speech_time_s": 45,
            "verbosity": "default",
            "follow_up_questions": True
        },
        "flow": "interview",
        "voice": {
            "provider": "coqui",
            "preset_id": "p226",
            "voice_id": "p226",
            "speed": 0.95,
            "pitch": -0.5,
            "style": "professional",
            "override": False
        },
        "moderation": {
            "enabled": True,
            "level": "moderate"
        },
        "should_tts": True,
        "metadata": {
            "source_template_id": "technical-interviewer"
    }


}

DEFAULT_CONFIG = {
    "display_name": "New Assistant",
    "slug": "new-assistant",
    "greeting": "Hello! How can I help you today?",
    "system_prompt": "ROLE: You are a capable, calm AI assistant in a real-time voice conversation.\nOBJECTIVE: Understand intent, provide accurate help, and keep it voice-friendly.\n\nCRITICAL GROUNDING RULE:\n- Base EVERY response strictly on the user's latest utterance\n- Do NOT assume any topic or domain\n- If the user says a single word, respond directly about that word\n- Adapt dynamically to whatever the user discusses\n\nCONVERSATION RULES:\n- Keep responses under 45 seconds when spoken\n- Use short, clear sentences; avoid rambling\n- Acknowledge the user's message before answering\n- Focus on the specific question; avoid tangents unless asked\n\nTURN-TAKING RULES:\n- Produce exactly ONE response per user utterance\n- After you speak, WAIT for the next user transcript from the system\n- Never simulate user replies or chain multiple turns\n\nWAITING BEHAVIOR:\n- Remain silent after your response until a new transcript arrives\n\nOUTPUT CONSTRAINTS:\n- Voice-first phrasing; no markdown/emojis; concise and structured",
    "examples": [],
    "behavior": {
        "max_speech_time_s": 45,
        "verbosity": "default",
        "follow_up_questions": False
    },
    "voice": {
        "provider": "coqui",
        "preset_id": "p225", # Default to Female Soft
        "voice_id": "p225",
        "speed": 1.0,
        "pitch": 0.0,
        "style": None,
        "override": False
    },
    "moderation": {
        "enabled": True,
        "level": "moderate"
    },
    "should_tts": True,
    "flow": "interview",
    "metadata": {
        "source_template_id": None
    }
}

class PersonaBuilder:
    def __init__(self):
        self.providers = get_providers()
        self.llm = self.providers.get('llm') 

    def generate_config(self, description_text):
        # Sync wrapper if needed, but we rely on async mostly now
        pass

    async def a_generate_config(self, description_text, current_voice=None, template_id=None):
        """
        Async generation using strict system prompt and schema validation.
        Returns tuple: (config_dict, raw_llm_output, token_usage_dict)
        
        If template_id matches a preset, return the preset immediately (fast, reliable).
        Otherwise, use LLM generation (slower, may fail).
        """
        # PRIORITY 1: Check for hardcoded preset templates
        if template_id and template_id in PRESET_TEMPLATES:
            logger.info(f"Using hardcoded preset template: {template_id}")
            import copy
            preset_config = copy.deepcopy(PRESET_TEMPLATES[template_id])
            
            # If user has custom voice settings, merge them (respect user's voice choice)
            if current_voice and not current_voice.get('override', False):
                preset_config['voice'] = {**preset_config['voice'], **current_voice}
            # Normalize textual fields for accuracy (convert any \n sequences to real newlines)
            preset_config = self._normalize_text_fields(preset_config)
            return preset_config, f"PRESET:{template_id}", {"total": 0}
        
        # PRIORITY 2: For custom descriptions without template_id, use LLM
        logger.info(f"No preset match for template_id={template_id}, using LLM generation")
        
        prompt = f"Description: {description_text}"
        if current_voice:
             prompt += f"\nCurrent Voice Settings: {json.dumps(current_voice)}"
        if template_id:
             prompt += f"\nTemplate ID: {template_id}"

        messages = [
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            if hasattr(self.llm, 'is_mock') and self.llm.is_mock:
                return self._heuristic_fallback(description_text), "MOCK_OUTPUT", {"total": 0}
                
            response = await self.llm.generate_response(messages, STRICT_SYSTEM_PROMPT)
            text = response.get('text', '{}')
            tokens = response.get('usage', {})
            
            # extract json - handle both clean and markdown-wrapped responses
            json_str = text.strip()
            
            # Remove markdown code blocks if present
            if json_str.startswith('```'):
                # Find the first { after the opening ```
                start = json_str.find('{')
                # Find the last } before the closing ```
                end = json_str.rfind('}')
                if start != -1 and end != -1:
                    json_str = json_str[start:end+1]
            else:
                # Just extract JSON object
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end > 0:
                    json_str = json_str[start:end]
            
            if not json_str:
                logger.error(f"No JSON found in LLM response: {text[:500]}")
                raise ValueError("No JSON found in LLM response")
            
            try:
                # Use strict=False to be more lenient with JSON parsing
                config = json.loads(json_str, strict=False)
                validated_config = self.validate_and_fix(config)
                # Normalize textual fields post-parse to ensure real newlines
                normalized = self._normalize_text_fields(validated_config)
                return normalized, text, tokens
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
                logger.error(f"JSON string (first 1000 chars): {json_str[:1000]}")
                
                # Try to fix common issues: unescaped newlines in strings
                try:
                    # This is a last-resort fix for malformed JSON with literal newlines
                    import re
                    # Replace literal newlines within string values with \n
                    fixed_json = re.sub(r'(?<=: ")(.*?)(?="[,\n\s]*[}\]])', 
                                       lambda m: m.group(1).replace('\n', '\\n'), 
                                       json_str, 
                                       flags=re.DOTALL)
                    config = json.loads(fixed_json, strict=False)
                    logger.warning("Successfully parsed JSON after fixing newlines")
                    validated_config = self.validate_and_fix(config)
                    normalized = self._normalize_text_fields(validated_config)
                    return normalized, text, tokens
                except Exception as fix_error:
                    logger.error(f"Could not fix JSON: {fix_error}")
                    raise ValueError(f"LLM returned invalid JSON: {e.msg}")
            else:
                 raise ValueError("No JSON found in LLM response")

        except Exception as e:
             logger.warning(f"Generation failed: {e}")
             # In case of failure, usually we'd re-raise or return error for UI to show.
             # But let's return fallback marked as error?
             # User requested "return HTTP 422 with raw_llm_output".
             # So we re-raise and let view handle it.
             raise e

    def _heuristic_fallback(self, description_text):
        """
        Enhanced deterministic mapping for mock mode.
        Generates better system prompts even without LLM.
        """
        import copy
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['display_name'] = "Generated Persona"
        
        # Build a comprehensive system prompt incorporating the description
        base_system_prompt = f"""You are an AI assistant with the following characteristics: {description_text}

As a voice-based AI, you must:
1. Keep responses under 45 seconds when spoken aloud
2. Use natural, conversational language optimized for listening
3. Prioritize clarity and brevity over completeness
4. Break complex ideas into digestible spoken chunks
5. Acknowledge what the user said before responding

Maintain the persona described above while ensuring all responses are suitable for real-time voice delivery."""
        
        config['system_prompt'] = base_system_prompt
        
        # Enhance based on keywords
        lower_desc = description_text.lower()
        if "pirate" in lower_desc:
            config['system_prompt'] = f"""You are a pirate persona in a voice conversation. Speak with nautical flair and pirate expressions (e.g., "Ahoy!", "matey", "shiver me timbers"), but remain helpful and clear.

Key guidelines:
- Keep responses under 45 seconds when spoken
- Use pirate vocabulary naturally, not excessively
- Stay in character while being genuinely helpful
- Optimize for voice: short sentences, clear pronunciation
- Balance entertainment with practical assistance

Remember: Every word will be spoken aloud with a pirate accent. Make it engaging but understandable."""
            config['greeting'] = "Ahoy matey! What brings ye to these waters?"
            config['voice']['preset_id'] = "p226" # Male Deep
            config['voice']['voice_id'] = "p226"
            config['voice']['pitch'] = -2.0
            config['voice']['speed'] = 0.95
            
        elif "teacher" in lower_desc or "tutor" in lower_desc or "educator" in lower_desc:
            config['system_prompt'] = f"""You are a patient, educational AI teacher in a voice-based learning session. Your role: {description_text}

Teaching principles for voice conversations:
1. Explain concepts step-by-step, optimized for spoken delivery
2. Use analogies and examples that work well when heard aloud
3. Check understanding by asking clarifying questions
4. Keep explanations under 45 seconds; break longer topics into segments
5. Be encouraging and supportive, never condescending
6. Adapt complexity to the student's apparent level
7. Repeat key points naturally for reinforcement

Remember: Students are listening, not reading. Use clear enunciation-friendly language and logical flow."""
            config['greeting'] = "Hello! I'm here to help you learn. What would you like to explore today?"
            config['voice']['preset_id'] = "p225"
            config['voice']['voice_id'] = "p225"
            config['voice']['speed'] = 0.90  # Slower for clarity
            config['behavior']['follow_up_questions'] = True
            
        elif "coach" in lower_desc or "mentor" in lower_desc:
            config['system_prompt'] = f"""You are an empathetic coach/mentor: {description_text}

Coaching approach for voice sessions:
1. Listen actively—acknowledge feelings and experiences before advising
2. Ask powerful, open-ended questions to promote reflection
3. Offer guidance, not directives; empower don't prescribe
4. Keep responses conversational and warm (under 45 seconds)
5. Use supportive, non-judgmental language
6. Validate emotions authentically
7. Pace the conversation—allow space for thinking

Voice considerations: Use a calm, measured tone. Pause points matter. Sound genuinely present."""
            config['greeting'] = "Welcome! I'm here to support you. What's on your mind today?"
            config['voice']['style'] = "calm"
            config['voice']['speed'] = 0.88
            config['behavior']['follow_up_questions'] = True
            
        elif "technical" in lower_desc or "engineer" in lower_desc or "expert" in lower_desc:
            config['system_prompt'] = f"""You are a technical expert: {description_text}

Technical communication for voice:
1. Use precise terminology, but define jargon when first introduced
2. Explain trade-offs and reasoning, not just solutions
3. Structure complex answers: "There are three key points..." then enumerate
4. Ask clarifying questions before deep-diving
5. Keep initial responses under 45 seconds; offer to elaborate if needed
6. Admit uncertainty rather than speculate
7. Match technical depth to the user's apparent level

Voice optimization: Technical content is harder to absorb aurally. Use extra structure and pacing."""
            config['greeting'] = "Hi! Ready to dive into some technical problem-solving?"
            config['voice']['preset_id'] = "p226"
            config['voice']['voice_id'] = "p226"
            config['behavior']['verbosity'] = "high"
            config['behavior']['max_speech_time_s'] = 60  # Technical topics may need more time
            
        return config

    def validate_and_fix(self, config):
        try:
            jsonschema.validate(instance=config, schema=PERSONA_SCHEMA)
            return config
        except jsonschema.ValidationError as e:
            logger.warning(f"Schema validation failed: {e.message}. Attempting to fix...")
            # Simple fix: merge with default to ensure missing fields exist
            fixed = DEFAULT_CONFIG.copy()
            
            # Recursive merge or shallow? Schema is nested. 
            # Smart merge required.
            
            # 1. Top level
            for k, v in config.items():
                if k in fixed and isinstance(fixed[k], dict) and isinstance(v, dict):
                    fixed[k].update(v)
                else:
                    fixed[k] = v
            
            # Re-validate
            try:
                jsonschema.validate(instance=fixed, schema=PERSONA_SCHEMA)
                return fixed
            except jsonschema.ValidationError as e2:
                raise ValueError(f"Could not fix schema error: {e2.message}")

    def _normalize_text_fields(self, config: dict) -> dict:
        """Convert any literal backslash-escaped newlines ("\\n") to real newlines in
        key user-facing text fields to maximize LLM accuracy and readability.
        Applies to: system_prompt, greeting, examples[*].text
        """
        import copy
        cfg = copy.deepcopy(config)

        def _norm(s: str) -> str:
            if not isinstance(s, str):
                return s
            # If the text contains literal "\\n", replace with actual newline.
            return s.replace("\\n", "\n")

        if 'system_prompt' in cfg:
            cfg['system_prompt'] = _norm(cfg['system_prompt'])
        if 'greeting' in cfg:
            cfg['greeting'] = _norm(cfg['greeting'])
        if 'examples' in cfg and isinstance(cfg['examples'], list):
            new_examples = []
            for ex in cfg['examples']:
                if isinstance(ex, dict) and 'text' in ex:
                    ex = {**ex, 'text': _norm(ex['text'])}
                new_examples.append(ex)
            cfg['examples'] = new_examples
        return cfg
