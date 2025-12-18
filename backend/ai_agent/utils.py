from typing import Dict, List, Any

def _is_followup_query(current_text: str, recent_history: List[Dict[str, str]]) -> bool:
    """
    Detect if current query is a brief follow-up to a previous question.
    Examples: "A simple example", "Yes", "Tell me more", "What about X?"
    """
    if not current_text:
        return False
    current_lower = current_text.lower().strip()
    
    # Heuristics: short queries that look like follow-ups
    followup_patterns = [
        r"^(a|an|the)?\s+\w+\s+example",  # "a simple example"
        r"^yes\.?$", r"^no\.?$",
        r"^ok\.?$|^okay\.?$",
        r"^tell me more",
        r"^what (about|if)",
        r"^how about",
        r"^explain",
        r"^details?",
        r"^example",
    ]
    
    import re
    is_short = len(current_text.split()) <= 4  # 4 words or less
    looks_like_followup = any(re.match(pat, current_lower) for pat in followup_patterns)
    
    if not (is_short and looks_like_followup):
        return False
    
    # Check if previous message exists and is a substantial question
    for msg in reversed(recent_history):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            prev_text = msg["content"].strip()
            # Previous was substantial (10+ chars) and likely a question
            has_question_mark = "?" in prev_text
            is_substantial = len(prev_text) > 10
            return is_substantial or has_question_mark
    
    return False

def build_prompt(
    persona_config: Dict[str, Any],
    history: List[Dict[str, str]],
    latest_transcript: str = "",
    max_tokens: int = 1500,
    mode: str = "json"  # "json" | "text" - controls output format instruction
) -> List[Dict[str, str]]:
    """
    Constructs prompt with rolling history and simple token budget management.
    mode="json": adds JSON output instruction (for evaluation/structured responses)
    mode="text": omits JSON instruction (for simple Q&A generation)
    
    CRITICAL: Keeps history short (4-6 messages max) to ensure the latest user
    utterance dominates the context and prevents topic drift.
    
    NEW: If current turn is a brief follow-up, includes prior question context.
    NEW: Adds brief-answer guidance for initial responses (detailed on follow-up).
    """
    messages = []
    
    # 1. System Prompt with explicit grounding reminder
    system_text = persona_config.get("system_prompt", "You are a helpful AI assistant.")
    # Normalize any literal backslash-escaped newlines for legacy records
    if isinstance(system_text, str) and "\\n" in system_text:
        system_text = system_text.replace("\\n", "\n")
    
    # Append Output Format Instruction only when mode="json"
    if mode == "json":
        system_text += "\n\nProvide your response as a valid JSON object with keys: 'text' (string) and 'should_tts' (boolean)."
    else:
        # For text mode (regular Q&A): add brief-answer guidance
        system_text += "\n\n[RESPONSE GUIDELINE] For your initial response: Provide a clear, concise answer (2-3 sentences for direct questions). If the user asks for more detail or examples, then expand with comprehensive explanation. Prioritize clarity and directness."

    messages.append({"role": "system", "content": system_text})
    
    # 2. Few-shot Examples - FILTER OUT DOMAIN-SPECIFIC ONES
    # Only include examples if they're neutral and multi-domain
    examples = persona_config.get("examples", [])
    filtered_examples = []
    for ex in examples:
        # New schema: {"role": "user"|"assistant", "text": "..."}
        if "role" in ex and "text" in ex:
            # Skip database/query-specific examples unless explicitly wanted
            text_lower = ex["text"].lower()
            if "database" in text_lower or "query" in text_lower or "sql" in text_lower:
                # Only include if user's latest transcript mentions these
                if latest_transcript and any(kw in latest_transcript.lower() for kw in ["database", "query", "sql"]):
                    filtered_examples.append({"role": ex["role"], "content": ex["text"]})
            else:
                filtered_examples.append({"role": ex["role"], "content": ex["text"]})
        # Legacy schema: {"user": "...", "assistant": "..."}
        elif "user" in ex:
            text_lower = ex["user"].lower()
            if not any(kw in text_lower for kw in ["database", "query", "sql"]):
                filtered_examples.append({"role": "user", "content": ex["user"]})
        elif "assistant" in ex:
            text_lower = ex["assistant"].lower()
            if not any(kw in text_lower for kw in ["database", "query", "sql"]):
                filtered_examples.append({"role": "assistant", "content": ex["assistant"]})
    
    messages.extend(filtered_examples)
            
    # 3. Conversation History - REDUCED to 4-6 messages (2-3 turns)
    # This ensures the latest user utterance has maximum influence
    recent_history = history[-6:] if len(history) > 6 else history
    
    # Check if current is a follow-up: if so, inject prior context
    is_followup = _is_followup_query(latest_transcript, recent_history)
    if is_followup:
        # Find the last substantial user question (2+ messages back)
        for i in range(len(recent_history) - 2, -1, -1):
            if recent_history[i].get("role") == "user":
                prior_question = recent_history[i].get("content", "").strip()
                if prior_question and len(prior_question) > 10:
                    # Inject a meta-note before the history
                    inject_msg = {
                        "role": "system",
                        "content": f"[Context: The user is asking for a follow-up. Their previous main question was: '{prior_question}' Please provide a response related to that prior topic.]"
                    }
                    messages.append(inject_msg)
                    break
    
    # Simple token estimation (1 token approx 4 chars)
    # If budget exceeded, trim further from top
    def estimate_tokens(msgs):
        return sum(len(m.get('content', '')[:5000]) for m in msgs) / 4  # cap per-message at 5000 chars

    current_tokens = estimate_tokens(messages)
    
    final_history = []
    for msg in reversed(recent_history):
        content = msg.get('content', '')[:5000]  # Safeguard: truncate very long messages
        msg_tokens = len(content) / 4
        if current_tokens + msg_tokens > max_tokens:
            break
        final_history.insert(0, {"role": msg.get("role", "user"), "content": content})
        current_tokens += msg_tokens
        
    messages.extend(final_history)
    
    # 4. Latest User Input (if provided) - THIS IS THE PRIMARY SIGNAL
    if latest_transcript and latest_transcript.strip():
        # Safeguard: truncate extremely long transcripts
        truncated = latest_transcript.strip()[:2000]
        messages.append({"role": "user", "content": truncated})
        
    return messages
