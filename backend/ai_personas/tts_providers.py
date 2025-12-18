from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator, Optional, List
import os
import io
import asyncio
import logging
import tempfile
import time
import shutil
import uuid
import threading
from pathlib import Path

from django.conf import settings
from .audio_utils import normalize_to_48k_mono_pcm, pcm_bytes_to_frames

logger = logging.getLogger(__name__)

# Global flag to track if we've already warned about missing TTS
_tts_init_warned = False

# Thread-safe TTS model pre-warming
_warmup_thread = None
_warmup_done = False


def _warmup_tts_background():
    """Initialize Coqui TTS model in background thread on startup."""
    global _warmup_done
    
    logger.info("🔥 Pre-warming Coqui TTS model in background...")
    try:
        # Import here to avoid blocking module load
        from TTS.api import TTS
        
        model_name = os.environ.get("COQUI_MODEL_NAME", "tts_models/en/vctk/vits")
        gpu = os.environ.get("COQUI_GPU", "true").lower() == "true"
        
        logger.info(f"Loading TTS model: {model_name} (GPU={gpu})")
        
        # This initialization takes 10-20s on first run
        # Running in background avoids blocking first WebRTC connection
        tts = TTS(model_name, progress_bar=False, gpu=gpu)
        
        logger.info("✅ Coqui TTS model pre-warming complete")
        _warmup_done = True
        
        # Cache it globally so first user gets instant synthesis
        import ai_personas.tts_providers as tts_mod
        tts_mod.CoquiTTSProvider._tts_instance = tts
        
    except Exception as e:
        logger.warning(f"⚠️ TTS pre-warm failed (will initialize on first use): {e}")
        _warmup_done = True  # Mark as done so we don't try again


# Start background warmup on module import (if TTS is configured and available)
if os.environ.get("TTS_PROVIDER", "coqui").lower() == "coqui":
    try:
        import TTS
        _warmup_thread = threading.Thread(target=_warmup_tts_background, daemon=True)
        _warmup_thread.start()
        logger.info("📦 Background TTS warmup started")
    except ImportError:
        logger.debug("TTS package not available; skipping background warmup")

class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """
        Convert text to audio bytes (WAV/PCM).
        Returns the full audio buffer normalized to 48kHz mono 16-bit PCM.
        """
        pass

    @abstractmethod
    async def stream(self, text: str, voice_config: Dict[str, Any]) -> AsyncIterator[bytes]:
        """
        Yields small PCM chunks (e.g. 20ms) suitable for aiortc.
        """
        pass

class MockTTS(TTSProvider):
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        logger.info(f"MockTTS synthesizing: {text[:30]}...")
        # Create a simple sine wave or silent WAV
        import wave
        with io.BytesIO() as bio:
            with wave.open(bio, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                # Generate 1 second of silence
                data = bytearray(48000 * 2) 
                wav.writeframes(data)
            return bio.getvalue()

    async def stream(self, text: str, voice_config: Dict[str, Any]) -> AsyncIterator[bytes]:
        # Yield silence chunks
        chunk_size = 960 * 2 # 20ms at 48kHz 16-bit mono
        for _ in range(50): # 1 second
            yield b'\x00' * chunk_size
            await asyncio.sleep(0.02)

class CoquiTTSProvider(TTSProvider):
    _semaphore = asyncio.Semaphore(1) # Limit concurrent synthesis to avoid CPU starvation
    _tts_instance = None
    _model_lock = asyncio.Lock() # Lock for async init
    _sync_lock = threading.Lock() # Lock for sync init
    
    def __init__(self):
        self.model_name = os.environ.get("COQUI_MODEL_NAME", "tts_models/en/vctk/vits")
        self.gpu = os.environ.get("COQUI_GPU", "true").lower() == "true"
        # Increase default timeout to 60s for initial model download/load
        self.timeout_ms = int(os.environ.get("TTS_TIMEOUT_MS", "60000"))
        
        # Verify ffmpeg on init
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found! CoquiTTSProvider will likely fail to normalize audio.")

    def _get_tts(self):
        """
        Singleton access to TTS instance with lazy loading.
        Uses a threading lock to be thread-safe during sync initialization.
        """
        if CoquiTTSProvider._tts_instance:
            return CoquiTTSProvider._tts_instance
            
        with CoquiTTSProvider._sync_lock:
            # Double check inside lock
            if CoquiTTSProvider._tts_instance:
                return CoquiTTSProvider._tts_instance
                
            logger.info(f"Initializing Coqui TTS with model {self.model_name} (GPU={self.gpu})")
            
            # --- Windows eSpeak Fix ---
            if os.name == 'nt' and not os.environ.get('PHONEMIZER_ESPEAK_PATH'):
                # Try to locate espeak automatically
                possible_paths = [
                    r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
                    r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
                    r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        logger.info(f"Found eSpeak at {p}, setting PHONEMIZER_ESPEAK_PATH")
                        os.environ['PHONEMIZER_ESPEAK_PATH'] = p
                        break
            # --------------------------

            try:
                from TTS.api import TTS
                # This can be slow (~10-20s for model load)
                CoquiTTSProvider._tts_instance = TTS(self.model_name, progress_bar=False, gpu=self.gpu)
                logger.info("Coqui TTS initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Coqui TTS: {e}")
                raise
            
        return CoquiTTSProvider._tts_instance

    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """
        Robust synthesis with concurrency limit, timeout, and retry.
        """
        async with self._semaphore:
            # Increase attempt count for robust model loading on first run
            for attempt in range(2): 
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(self._synthesize_sync, text, voice_config),
                        timeout=self.timeout_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"TTS synthesis timed out (attempt {attempt+1})")
                    if attempt == 1: raise
                except Exception as e:
                    logger.error(f"TTS synthesis error (attempt {attempt+1}): {e}", exc_info=True)
                    if attempt == 1: 
                        # Fallback to MockTTS if Coqui dies completely? 
                        # The factory usually handles fallback if provider init fails, 
                        # but here runtime failure -> return silence or error?
                        # Returning error allows Orchestrator to decide (e.g. text fallback)
                        raise

    def _synthesize_sync(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        logger.info(f"Starting synchronous synthesis for text='{text[:20]}...'")
        try:
            tts = self._get_tts()
        except Exception as e:
            logger.error(f"Failed to get TTS instance: {e}")
            raise

        speaker = voice_config.get("voice_id")
        rate = voice_config.get("rate", 1.0)
        
        logger.info(f"Requested speaker='{speaker}' speed={rate}")

        # Handle multi-speaker models
        try:
            is_multi = tts.is_multi_speaker
            logger.info(f"Model is_multi_speaker={is_multi}")
            
            if is_multi:
                # Debug available speakers
                try:
                    speakers = tts.speakers
                    logger.info(f"Available speakers (first 5): {speakers[:5] if speakers else 'None'}")
                except Exception as e:
                    logger.warning(f"Failed to access tts.speakers: {e}")
                    speakers = []

                if not speaker:
                    if speakers:
                        speaker = speakers[0]
                        logger.info(f"No speaker provided, using default: {speaker}")
                
                # Verify speaker exists
                if speaker and speakers and speaker not in speakers:
                    logger.warning(f"Speaker {speaker} not found in model. Falling back to {speakers[0]}")
                    speaker = speakers[0]
        except Exception as e:
            logger.error(f"Error checking speaker properties: {e}", exc_info=True)
            # Proceed cautiously or default
            if not speaker:
                speaker = None

        logger.info(f"Final parameters: speaker='{speaker}' speed={rate}")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            # This operation is blocking and CPU intensive
            start = time.time()
            
            # Attempt synthesis
            # Note: We do NOT pass speed/rate here anymore because it was causing KeyErrors.
            # We will handle speed and pitch via post-processing (FFmpeg)
            tts.tts_to_file(text=text, speaker=speaker, file_path=tmp_path)
            
            dur = time.time() - start
            logger.info(f"Coqui raw synthesis took {dur:.2f}s")
            
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()
                
            # Normalize first (to 48k mono)
            # This is important before applying effects because our effect logic assumes 48k base
            pcm_bytes_normalized = normalize_to_48k_mono_pcm(wav_bytes)
            
            if not pcm_bytes_normalized:
                 raise ValueError("Audio normalization failed")
            
            # Now apply effects (Speed / Pitch)
            # We need to wrap it in a WAV container again for ffmpeg to read it as input? 
            # normalize_to_48k_mono_pcm returns raw PCM (s16le). 
            # apply_audio_effects expects WAV or recognizable input.
            # Actually, normalize_to_48k_mono_pcm returns raw bytes.
            # Let's verify apply_audio_effects input handling.
            # It uses `ffmpeg -i pipe:0`. FFmpeg might not guess raw PCM without args.
            # So, apply_audio_effects should probably take PCM and args, or we wrap it.
            # Easier: pass the ORIGINAL wav_bytes to apply_effects first?
            # BUT original wav_bytes might have different sample rate (22050).
            # The logic in apply_audio_effects assumes 48000 base_sr because of our math.
            
            # Better approach:
            # 1. Normalize RAW wav from Coqui to 48k WAV (not raw PCM).
            # 2. Apply effects.
            # 3. Return raw PCM.
            
            # However `normalize_to_48k_mono_pcm` currently returns raw PCM s16le.
            # Let's modify apply_audio_effects to accept s16le raw input if we specify -f s16le -ar 48000 -ac 1
            # OR just wrap it in a BytesIO WAV before passing.
            
            # Let's wrap PCM in WAV for the effects function
            import io
            import wave
            from .audio_utils import apply_audio_effects

            with io.BytesIO() as intermediate_wav:
                with wave.open(intermediate_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(pcm_bytes_normalized)
                formatted_wav = intermediate_wav.getvalue()
                
            # Apply effects
            # Convert semitones to multiplier: 2^(semitones/12)
            pitch_semitones = float(voice_config.get("pitch", 0.0))
            pitch_multiplier = 2 ** (pitch_semitones / 12.0)
            
            processed_wav = apply_audio_effects(formatted_wav, rate=rate, pitch=pitch_multiplier)
            
            # processed_wav is a WAV file bytes (checked apply_audio_effects implementation - output -f wav).
            # We need to return raw PCM for the provider contract? 
            # The docstring says "returns ... normalized to 48kHz mono 16-bit PCM".
            # So we strip the header again.
            
            # Simple header strip for 44-byte WAV? Or run through normalize again just to be safe/lazy?
            return normalize_to_48k_mono_pcm(processed_wav)
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def stream(self, text: str, voice_config: Dict[str, Any]) -> AsyncIterator[bytes]:
        """
        Pre-buffer mode: Synthesize full clip -> Split -> Yield.
        """
        try:
            full_audio_pcm = await self.synthesize(text, voice_config)
            frames = pcm_bytes_to_frames(full_audio_pcm, frame_ms=20, sample_rate=48000)
            
            for frame in frames:
                yield frame
                # Yield control to event loop to simulate stream & preventing blocking
                await asyncio.sleep(0) 
        except Exception as e:
            logger.error(f"Stream synthesis failed: {e}")
            # Fallback to silence if stream fails mid-way?
            # Or just stop.
            pass

class ProviderFactory:
    _instance = None
    
    @classmethod
    def get_tts_provider(cls) -> TTSProvider:
        mode = os.environ.get("AI_MODE", "mock").lower()
        provider_name = os.environ.get("TTS_PROVIDER", "coqui")
        
        # Determine effective provider
        # If AI_MODE=live, we might map to 'openai' or 'azure' etc, but for now fallback to mock is fine if not imp
        
        if provider_name == "mock_simple":
             return MockTTS()
        
        # Check if we should use Coqui
        if provider_name == "coqui":
            # Check for TTS package availability
            try:
                import TTS
            except ImportError:
                logger.warning("The 'TTS' python package is not installed or failed to import. Falling back to MockTTS.")
                return MockTTS()

            # We can cache the provider instance
            if not getattr(cls, "_coqui_instance", None):
                cls._coqui_instance = CoquiTTSProvider()
            return cls._coqui_instance
            
        return MockTTS()

