import asyncio
import os
import json
import logging
import numpy as np
from .interfaces import STTProvider, LLMProvider, TTSProvider
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MockSTT(STTProvider):
    """
    MockSTT for development/testing: returns canned responses regardless of actual audio.
    
    THIS IS EXPECTED BEHAVIOR IN DEVELOPMENT MODE.
    
    Your voice IS being captured and processed by the backend:
    - Audio frames are received and buffered
    - Voice activity detection (VAD) identifies speech
    - Utterances are detected (silence triggers transcription)
    - But instead of real speech-to-text, MockSTT returns pre-written responses
    
    To enable REAL speech-to-text:
    1. Option A: Switch to SimpleKeywordSTT (recognizes some common words)
    2. Option B: Integrate Groq Whisper API or other STT provider
    3. Option C: Use Google Cloud Speech-to-Text or Azure Speech Services
    
    For now, the canned responses allow you to test the full conversation flow.
    """
    def __init__(self):
        self._transcription_count = 0
        self.TRANSCRIPTIONS = [
            "I'm interested in learning more about this.",
            "That sounds fascinating, tell me more.",
            "What are some examples you can provide?",
            "How does that work exactly?",
            "I'd like to understand this better.",
            "Can you explain that in more detail?",
            "That's interesting, please continue.",
            "What should I focus on?",
            "I appreciate the information.",
            "How can I apply this knowledge?",
        ]
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        logger.info(f"🎤 MockSTT.transcribe: received {len(audio_bytes)} bytes of audio")
        logger.info(f"   ⚠️  MOCK MODE: Returning canned response (not real speech-to-text)")
        logger.info(f"   To use real STT, set STT_PROVIDER=keyword or integrate a real STT API")
        
        # Analyze audio to at least show if speech was detected
        try:
            # Convert bytes to audio samples (assuming 16-bit PCM, 48kHz)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_data) > 0:
                energy = np.sqrt(np.mean(audio_data**2))
                logger.info(f"   Audio energy: {energy:.0f} | Speech detected: {'Yes' if energy > 500 else 'Quiet'}")
        except Exception as e:
            logger.debug(f"   Could not analyze audio: {e}")
        
        result = self.TRANSCRIPTIONS[self._transcription_count % len(self.TRANSCRIPTIONS)]
        self._transcription_count += 1
        logger.info(f"   Canned response #{self._transcription_count}: '{result}'")
        return result


class SimpleKeywordSTT(STTProvider):
    """
    Simple keyword-based speech recognition for basic commands.
    Useful for testing when real STT is not available.
    
    Recognizes patterns in audio energy levels to guess common words.
    NOT ACCURATE - for demonstration purposes only.
    """
    def __init__(self):
        self.KEYWORDS = {
            "python": "I want to learn Python",
            "javascript": "I want to learn JavaScript",
            "database": "I need help with databases",
            "help": "I need some help",
            "yes": "Yes, please",
            "no": "No, I don't think so",
            "more": "Tell me more",
            "stop": "That's all for now",
        }
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        logger.info(f"🎤 SimpleKeywordSTT: received {len(audio_bytes)} bytes")
        
        try:
            # Very simple heuristic: analyze audio pattern
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_data) == 0:
                return "I'm listening"
            
            # Split into chunks and analyze patterns
            chunk_size = len(audio_data) // 10
            if chunk_size == 0:
                chunk_size = len(audio_data)
            
            energies = []
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                energy = np.sqrt(np.mean(chunk**2))
                energies.append(energy)
            
            # Simple pattern matching (very crude!)
            avg_energy = np.mean(energies) if energies else 0
            max_energy = np.max(energies) if energies else 0
            variance = np.var(energies) if len(energies) > 1 else 0
            
            logger.info(f"   Audio pattern - avg: {avg_energy:.0f}, max: {max_energy:.0f}, var: {variance:.0f}")
            
            # Very naive heuristic based on energy patterns
            if variance > 100000:  # High variance = more speech-like
                return "I'd like to know more"
            elif max_energy > 3000:  # Loud speech
                return "Yes, I understand"
            elif max_energy > 1500:  # Medium speech
                return "That's interesting"
            else:  # Quiet
                return "Can you speak up?"
                
        except Exception as e:
            logger.error(f"   Error in SimpleKeywordSTT: {e}")
            return "I didn't quite catch that"

class MockLLM(LLMProvider):
    async def generate_response(self, messages: list[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        logger.info(f"🧠 MockLLM.generate_response: received {len(messages)} messages")
        logger.info(f"   System prompt: '{system_prompt[:100]}...'")
        last_user_msg = messages[-1]['content'] if messages else ""
        logger.info(f"   Last user message: '{last_user_msg}'")
        
        # Simple echo/template logic
        if "hello" in last_user_msg.lower():
            text = "Hi! I am the mock AI. How can I help?"
        else:
            text = f"I heard you say: {last_user_msg}"
        
        logger.info(f"🧠 MockLLM.generate_response: returning text='{text}'")
            
        return {
            "text": text,
            "should_tts": True
        }

class MockTTS(TTSProvider):
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        logger.info(f"MockTTS: Synthesizing '{text}'")
        # Return a simple beep or silence wav header for valid audio
        # 44100Hz 16bit mono WAV header + some silence/noise
        # Minimal valid WAV file
        wav_header = b'\x52\x49\x46\x46\x24\x00\x00\x00\x57\x41\x56\x45\x66\x6d\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00\x64\x61\x74\x61\x00\x00\x00\x00'
        return wav_header + b'\x00' * 1000 # Short silence

def get_providers() -> Dict[str, Any]:
    mode = os.environ.get("AI_MODE", "mock").lower()
    stt_provider = os.environ.get("STT_PROVIDER", "sarvam").lower()  # Default to Sarvam AI
    
    # Select STT provider
    if mode == "live":
        # In live mode, try to use real STT
        if stt_provider == "sarvam":
            try:
                from .live_providers.stt import SarvamSTT
                stt = SarvamSTT()
                stt_name = "SarvamSTT"
                logger.info("✅ Using SarvamSTT (Sarvam AI speech-to-text)")
            except Exception as e:
                logger.warning("⚠️  Failed to load SarvamSTT: %s, falling back to MockSTT", e)
                stt = MockSTT()
                stt_name = "MockSTT"
        elif stt_provider in {"whisper", "local"}:
            try:
                from .live_providers.stt import WhisperLocalSTT
                stt = WhisperLocalSTT()
                stt_name = "WhisperLocalSTT"
                logger.info("✅ Using WhisperLocalSTT (FREE, offline, GPU-accelerated speech-to-text)")
            except Exception as e:
                logger.warning("⚠️  Failed to load WhisperLocalSTT: %s, falling back to MockSTT", e)
                stt = MockSTT()
                stt_name = "MockSTT"
        elif stt_provider == "groq":
            try:
                from .live_providers.stt import GroqWhisperSTT
                stt = GroqWhisperSTT()
                stt_name = "GroqWhisperSTT"
                logger.info("✅ Using GroqWhisperSTT (real speech-to-text via Groq)")
            except Exception as e:
                logger.warning("⚠️  Failed to load GroqWhisperSTT: %s, falling back to MockSTT", e)
                stt = MockSTT()
                stt_name = "MockSTT"
        elif stt_provider in {"speech", "speech_recognition", "google"}:
            try:
                from .live_providers.stt import SpeechRecognitionSTT
                stt = SpeechRecognitionSTT()
                stt_name = "SpeechRecognitionSTT"
                logger.info("✅ Using SpeechRecognitionSTT (Google via speech_recognition)")
            except Exception as e:
                logger.warning("⚠️  Failed to load SpeechRecognitionSTT: %s, falling back to MockSTT", e)
                stt = MockSTT()
                stt_name = "MockSTT"
        else:
            stt = MockSTT()
            stt_name = "MockSTT"
            logger.info("ℹ️  Using MockSTT (canned responses for testing)")
    else:
        # In mock mode, use mock STT
        stt = MockSTT()
        stt_name = "MockSTT"
        logger.info("ℹ️  Using MockSTT (canned responses for testing)")
    
    # Load TTS provider (same for both live and mock modes)
    from ai_personas.tts_providers import ProviderFactory
    tts = ProviderFactory.get_tts_provider()
    
    if mode == "live":
        # Import live Groq LLM when in live mode
        try:
            from .live_providers.groq_llm import GroqLLM
            llm = GroqLLM()
            logger.info(f"✅ Loaded live Groq LLM provider")
        except Exception as e:
            logger.error(f"❌ Failed to load GroqLLM: {e}, falling back to mock")
            llm = MockLLM()
        
        logger.info(f"🔧 Loaded providers: STT={stt_name}, LLM={type(llm).__name__}, TTS={type(tts).__name__}")
        return {"stt": stt, "llm": llm, "tts": tts}
    else:
        logger.info(f"🔧 Loaded providers: STT={stt_name}, LLM=MockLLM, TTS={type(tts).__name__} (MOCK MODE)")
        return {
            "stt": stt,
            "llm": MockLLM(),
            "tts": tts
        }
