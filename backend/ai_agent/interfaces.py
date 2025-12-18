from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to text."""
        pass

class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, messages: list[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """
        Generate a structured response.
        Returns JSON dict with keys like 'text', 'should_tts'.
        """
        pass

class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """Convert text to audio bytes."""
        pass
