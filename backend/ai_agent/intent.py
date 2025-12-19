import re
from typing import Dict, Any

INTENT_KEYWORDS = {
    'Question': [r"\?$", r"^how\b", r"^what\b", r"^why\b", r"^can you\b", r"^could you\b", r"^do you know\b"],
    'Command': [r"^show\b", r"^tell\b", r"^explain\b", r"^generate\b", r"^write\b", r"^help me\b"],
    'Affirmation': [r"^yes$", r"^yep$", r"^sure$", r"^correct$", r"^right$"],
    'Negation': [r"^no$", r"^nope$", r"^not really$", r"^incorrect$"],
    'SmallTalk': [r"^hi$", r"^hello$", r"^thanks?\b", r"^how are you\b"],
    'TopicSwitch': [r"switch topic", r"change topic", r"let's talk about", r"now about"],
}

TOPIC_MAP = {
    'python': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
    'javascript': ['javascript', 'js', 'react', 'next.js', 'nextjs', 'node', 'typescript'],
    'devops': ['docker', 'kubernetes', 'k8s', 'terraform', 'ci/cd', 'redis', 'nginx'],
    'web_development': ['frontend', 'backend', 'api', 'http', 'browser'],
}

def resolve_intent(text: str, memory: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s = (text or '').strip()
    lower = s.lower()
    tokens = [t for t in re.split(r"\s+", lower) if t]
    word_count = len(tokens)

    # Default
    intent_type = 'Unclear'
    for label, patterns in INTENT_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, lower):
                intent_type = label
                break
        if intent_type != 'Unclear':
            break

    # Topic guess by keywords
    topic = None
    for t, kws in TOPIC_MAP.items():
        if any(kw in lower for kw in kws):
            topic = t
            break

    # Confidence: simple heuristic on length and matched patterns
    confidence = 0.3
    if word_count >= 10:
        confidence += 0.3
    elif word_count >= 4:
        confidence += 0.2
    if intent_type != 'Unclear':
        confidence += 0.2
    confidence = min(1.0, max(0.0, confidence))

    requires_clarification = (intent_type == 'Unclear') or (word_count < 3)

    # Topic switch detection vs memory
    topic_changed = False
    if memory and topic and memory.get('current_topic') and topic != memory.get('current_topic'):
        topic_changed = True

    return {
        'intent_type': intent_type,
        'topic': topic,
        'confidence': confidence,
        'requires_clarification': requires_clarification,
        'topic_changed': topic_changed,
    }
