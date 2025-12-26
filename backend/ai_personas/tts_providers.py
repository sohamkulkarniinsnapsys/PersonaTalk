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

        # Default to XTTS v2 for Indian accent support
        model_name = os.environ.get("COQUI_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
        gpu = os.environ.get("COQUI_GPU", "true").lower() == "true"

        logger.info(f"Loading TTS model: {model_name} (GPU={gpu})")

        # Add timeout to background warmup (60 seconds max)
        # If it takes longer, we'll initialize on first use instead
        start_time = time.time()
        timeout_sec = 60
        
        # This initialization takes 10-20s on first run normally
        # If it's taking > 60s, something is wrong with the model or system
        try:
            tts = TTS(model_name, progress_bar=False, gpu=gpu)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Coqui TTS model pre-warming complete ({elapsed:.1f}s)")
            _warmup_done = True

            # Cache it globally so first user gets instant synthesis
            import ai_personas.tts_providers as tts_mod
            if not hasattr(tts_mod.CoquiTTSProvider, "_tts_instances"):
                tts_mod.CoquiTTSProvider._tts_instances = {}
            tts_mod.CoquiTTSProvider._tts_instances[model_name] = tts
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"⚠️ TTS pre-warm failed after {elapsed:.1f}s: {e}")
            _warmup_done = True  # Still mark as done so foreground doesn't wait forever

    except Exception as e:
        logger.warning(f"⚠️ TTS pre-warm failed: {e}")
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
    _tts_instances: dict[str, any] = {}
    _model_lock = asyncio.Lock() # Lock for async init
    _sync_lock = threading.Lock() # Lock for sync init
    _speaker_ref_cache: dict[str, Path] = {}
    
    def __init__(self):
        self.default_model_name = os.environ.get("COQUI_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
        self.gpu = os.environ.get("COQUI_GPU", "true").lower() == "true"
        self.gpu_fallback_enabled = os.environ.get("COQUI_GPU_FALLBACK", "true").lower() == "true"
        # CRITICAL: 180s timeout for XTTS v2 first-run (model load + synthesis)
        self.timeout_ms = int(os.environ.get("TTS_TIMEOUT_MS", "180000"))
        self._gpu_failed = False  # Track GPU failures for auto-fallback
        
        # Verify ffmpeg on init
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found! CoquiTTSProvider will likely fail to normalize audio.")

    def _normalize_model_name(self, model_name: str | None) -> str:
        if not model_name:
            return self.default_model_name
        if model_name == "xtts_v2":
            return "tts_models/multilingual/multi-dataset/xtts_v2"
        return model_name

    def _resolve_speaker_ref(self, speaker_ref: str | None) -> Path | None:
        if not speaker_ref:
            return None
        cached = self._speaker_ref_cache.get(speaker_ref)
        if cached and cached.exists():
            return cached
        path = Path(speaker_ref)
        if not path.is_absolute():
            path = Path(settings.BASE_DIR) / speaker_ref
        if not path.exists():
            raise FileNotFoundError(f"speaker_ref not found at {path}")
        self._speaker_ref_cache[speaker_ref] = path
        return path

    def _get_tts(self, model_name: str | None = None):
        """
        Singleton access to TTS instance per model with lazy loading.
        Uses a threading lock to be thread-safe during sync initialization.
        Does NOT block on background warmup - proceeds immediately if model needed.
        """
        normalized = self._normalize_model_name(model_name)
        if normalized in CoquiTTSProvider._tts_instances:
            return CoquiTTSProvider._tts_instances[normalized]
        
        # Don't wait for background warmup - it may load a different model
        # Just proceed with initialization immediately (the warmup helps first user, but shouldn't block subsequent users)
        global _warmup_done, _warmup_thread
        if _warmup_thread and not _warmup_done:
            # Quick check (100ms) to see if warmup is almost done
            _warmup_thread.join(timeout=0.1)
            if _warmup_done and normalized in CoquiTTSProvider._tts_instances:
                logger.info(f"📦 Using pre-warmed model: {normalized}")
                return CoquiTTSProvider._tts_instances[normalized]
            # If warmup not done or wrong model, proceed anyway (don't block)
            
        with CoquiTTSProvider._sync_lock:
            # Double check inside lock
            if normalized in CoquiTTSProvider._tts_instances:
                return CoquiTTSProvider._tts_instances[normalized]
                
            logger.info(f"Initializing Coqui TTS with model {normalized} (GPU={self.gpu})")
            
            # --- Windows eSpeak Fix ---
            if os.name == 'nt' and not os.environ.get('PHONEMIZER_ESPEAK_PATH'):
                # Try to locate espeak automatically
                possible_paths = [
                    r"C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",
                    r"C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
                    r"C:\\Program Files\\eSpeak NG\\espeak-ng.exe",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        logger.info(f"Found eSpeak at {p}, setting PHONEMIZER_ESPEAK_PATH")
                        os.environ['PHONEMIZER_ESPEAK_PATH'] = p
                        break
            # --------------------------

            try:
                from TTS.api import TTS
                import torch
                
                # Check GPU availability and VRAM
                use_gpu = self.gpu and not self._gpu_failed
                if use_gpu and torch.cuda.is_available():
                    try:
                        # Check VRAM before loading (XTTS v2 needs ~2GB)
                        vram_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        vram_free_gb = vram_free / (1024**3)
                        logger.info(f"GPU VRAM available: {vram_free_gb:.2f} GB")
                        
                        if vram_free_gb < 1.5:  # Less than 1.5GB free
                            logger.warning(f"Insufficient VRAM ({vram_free_gb:.2f}GB). Falling back to CPU.")
                            use_gpu = False
                            self._gpu_failed = True
                    except Exception as vram_err:
                        logger.warning(f"Could not check VRAM: {vram_err}. Attempting GPU anyway.")
                
                # This can be slow (60-120s for XTTS v2 model load on first run)
                logger.info(f"Loading TTS model with GPU={use_gpu}...")
                CoquiTTSProvider._tts_instances[normalized] = TTS(normalized, progress_bar=False, gpu=use_gpu)
                logger.info(f"✅ Coqui TTS initialized successfully (GPU={use_gpu}).")
                
            except RuntimeError as e:
                # Catch CUDA OOM and retry with CPU
                if "out of memory" in str(e).lower() and self.gpu_fallback_enabled and not self._gpu_failed:
                    logger.error(f"GPU OOM during TTS init: {e}")
                    logger.info("🔄 Retrying with CPU fallback...")
                    self._gpu_failed = True
                    try:
                        CoquiTTSProvider._tts_instances[normalized] = TTS(normalized, progress_bar=False, gpu=False)
                        logger.info("✅ Coqui TTS initialized successfully with CPU fallback.")
                    except Exception as cpu_err:
                        logger.error(f"CPU fallback also failed: {cpu_err}")
                        raise
                else:
                    logger.error(f"Failed to initialize Coqui TTS: {e}")
                    raise
            except Exception as e:
                logger.error(f"Failed to initialize Coqui TTS: {e}")
                raise
            
        return CoquiTTSProvider._tts_instances[normalized]

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
                    timeout_sec = self.timeout_ms / 1000.0
                    logger.error(
                        f"⏱️ TTS synthesis TIMEOUT after {timeout_sec}s (attempt {attempt+1}/2). "
                        f"Text length: {len(text)} chars. "
                        f"This indicates system resource exhaustion or slow GPU. "
                        f"Consider: (1) Increase TTS_TIMEOUT_MS, (2) Use CPU mode, (3) Reduce concurrent requests."
                    )
                    if attempt == 1: 
                        raise TimeoutError(
                            f"TTS synthesis exceeded {timeout_sec}s timeout. "
                            f"System may be overloaded or GPU VRAM exhausted."
                        )
                except Exception as e:
                    logger.error(f"TTS synthesis error (attempt {attempt+1}): {e}", exc_info=True)
                    if attempt == 1: 
                        # Fallback to MockTTS if Coqui dies completely? 
                        # The factory usually handles fallback if provider init fails, 
                        # but here runtime failure -> return silence or error?
                        # Returning error allows Orchestrator to decide (e.g. text fallback)
                        raise

    def _synthesize_sync(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        import time
        synthesis_start = time.time()
        logger.info(f"🎙️ Starting TTS synthesis: {len(text)} chars, text='{text[:20]}...'")
        model_name = voice_config.get("model")
        language = voice_config.get("language", "en")
        speaker_ref = voice_config.get("speaker_ref")
        speaker = voice_config.get("voice_id")
        # CRITICAL FIX: Coqui/XTTS don't support native speed control
        # Speed post-processing via FFmpeg causes unnatural audio
        # Always use natural 1.0x speed (ignore speed parameter from config)
        rate = 1.0

        try:
            resolved_speaker_ref = self._resolve_speaker_ref(speaker_ref)
        except FileNotFoundError as e:
            logger.error(str(e))
            raise

        try:
            tts = self._get_tts(model_name)
        except Exception as e:
            logger.error(f"Failed to get TTS instance: {e}")
            raise

        # Determine language support; drop language if model is not multilingual
        is_multilingual = bool(getattr(tts, "is_multi_lingual", False))
        if not is_multilingual and language:
            logger.info("Model is not multilingual; ignoring provided language parameter")
            language = None

        normalized_model = self._normalize_model_name(model_name)
        
        # CRITICAL: Detect XTTS models early (they require speaker_wav, not speaker IDs)
        is_xtts = 'xtts' in normalized_model.lower()
        
        logger.info(
            f"Requested model='{normalized_model}' "
            f"speaker='{speaker}' speaker_ref='{resolved_speaker_ref}' speed={rate} lang={language}"
        )

        # Handle multi-speaker models when no speaker_ref is provided
        # SKIP this logic for XTTS (it doesn't have predefined speakers)
        try:
            is_multi = getattr(tts, "is_multi_speaker", False)
            if is_multi and not resolved_speaker_ref and not is_xtts:
                try:
                    speakers = tts.speakers
                    logger.info(f"Available speakers (first 5): {speakers[:5] if speakers else 'None'}")
                except Exception as e:
                    logger.warning(f"Failed to access tts.speakers: {e}")
                    speakers = []

                if not speaker and speakers:
                    speaker = speakers[0]
                    logger.info(f"No speaker provided, using default: {speaker}")
                if speaker and speakers and speaker not in speakers:
                    logger.warning(f"Speaker {speaker} not found in model. Falling back to {speakers[0]}")
                    speaker = speakers[0]
        except Exception as e:
            logger.error(f"Error checking speaker properties: {e}", exc_info=True)
            if not speaker:
                speaker = None

        logger.info(
            f"Final parameters: model='{normalized_model}' "
            f"speaker='{speaker}' speaker_ref='{resolved_speaker_ref}' speed={rate} lang={language or 'none'}"
        )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # This operation is blocking and CPU intensive
            start = time.time()

            synth_kwargs = {"text": text, "file_path": tmp_path}
            
            # Detect XTTS models (use voice cloning, not speaker IDs)
            is_xtts = 'xtts' in normalized_model.lower()
            
            # XTTS v2 expects speaker_wav for voice cloning; does NOT support speaker IDs
            if is_xtts:
                if resolved_speaker_ref:
                    synth_kwargs["speaker_wav"] = str(resolved_speaker_ref)
                    logger.info(f"XTTS: Using voice cloning from {resolved_speaker_ref}")
                else:
                    # Use default LJSpeech sample (clear female English voice)
                    default_speaker = Path(settings.BASE_DIR) / "speaker_references" / "xtts_default.wav"
                    if default_speaker.exists():
                        synth_kwargs["speaker_wav"] = str(default_speaker)
                        logger.info(f"XTTS: Using default speaker reference: {default_speaker}")
                    else:
                        raise ValueError(
                            "XTTS requires speaker_wav for voice cloning. "
                            "No speaker_ref provided and default not found. "
                            f"Run: python download_xtts_default_speaker.py"
                        )
                # Do NOT pass speaker parameter to XTTS - it doesn't have predefined speakers
            else:
                # Non-XTTS models use speaker_ref for cloning OR speaker ID
                if resolved_speaker_ref:
                    synth_kwargs["speaker_wav"] = str(resolved_speaker_ref)
                elif speaker:
                    synth_kwargs["speaker"] = speaker
            
            if language:
                synth_kwargs["language"] = language
            temperature = voice_config.get("temperature")
            if temperature is not None:
                synth_kwargs["temperature"] = temperature

            tts.tts_to_file(**synth_kwargs)

            dur = time.time() - start
            logger.info(f"Coqui raw synthesis took {dur:.2f}s")

            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            # Normalize first (to 48k mono)
            pcm_bytes_normalized = normalize_to_48k_mono_pcm(wav_bytes)

            if not pcm_bytes_normalized:
                raise ValueError("Audio normalization failed")

            # Wrap PCM in WAV for effects
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
            pitch_semitones = float(voice_config.get("pitch", 0.0))
            pitch_multiplier = 2 ** (pitch_semitones / 12.0)

            # NOTE: Speed parameter removed - Coqui/XTTS don't support native speed control
            # Post-processing via FFmpeg atempo creates unnatural speech and adds latency
            # Instead, we keep natural speech by NOT applying rate changes
            processed_wav = apply_audio_effects(formatted_wav, rate=1.0, pitch=pitch_multiplier)

            final_pcm = normalize_to_48k_mono_pcm(processed_wav)
            
            total_time = time.time() - synthesis_start
            logger.info(f"✅ TTS synthesis complete: {total_time:.2f}s total (raw: {dur:.2f}s, processing: {total_time-dur:.2f}s)")
            
            return final_pcm

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
            logger.info(f"🎙️ Stream starting for text: {text[:80]}")
            full_audio_pcm = await self.synthesize(text, voice_config)
            logger.info(f"✅ Stream synthesized, splitting into frames")
            frames = pcm_bytes_to_frames(full_audio_pcm, frame_ms=20, sample_rate=48000)
            frame_count = 0
            for frame in frames:
                frame_count += 1
                yield frame
                # Yield control to event loop to simulate stream & preventing blocking
                await asyncio.sleep(0) 
            logger.info(f"✅ Stream complete: yielded {frame_count} frames")
        except Exception as e:
            logger.error(f"Stream synthesis failed: {e}")
            # Fallback to silence if stream fails mid-way?
            # Or just stop.
            pass

class ProviderFactory:
    _instance = None
    _coqui_instance = None
    _sarvam_instance = None
    
    @classmethod
    def get_tts_provider(cls) -> TTSProvider:
        mode = os.environ.get("AI_MODE", "mock").lower()
        provider_name = os.environ.get("TTS_PROVIDER", "coqui").lower()
        
        # Determine effective provider
        # Options: "mock_simple", "coqui", "sarvam"
        
        if provider_name == "mock_simple":
            return MockTTS()
        
        # Check if we should use Sarvam AI TTS
        if provider_name == "sarvam":
            try:
                from ai_agent.live_providers.tts import SarvamTTS
                
                # Cache the instance
                if not cls._sarvam_instance:
                    cls._sarvam_instance = SarvamTTS()
                    logger.info("✅ Initialized Sarvam AI TTS provider (cached)")
                return cls._sarvam_instance
            except ImportError as e:
                logger.error(f"❌ Sarvam TTS import failed: {e}. Falling back to MockTTS.")
                return MockTTS()
            except RuntimeError as e:
                logger.error(f"❌ Sarvam TTS initialization failed: {e}. Ensure SARVAM_API_KEY is set. Falling back to MockTTS.")
                return MockTTS()
        
        # Check if we should use Coqui XTTS
        if provider_name == "coqui":
            # Check for TTS package availability
            try:
                import TTS
            except ImportError:
                logger.warning("The 'TTS' python package is not installed or failed to import. Falling back to MockTTS.")
                return MockTTS()

            # Cache the provider instance
            if not cls._coqui_instance:
                cls._coqui_instance = CoquiTTSProvider()
                logger.info("✅ Initialized Coqui XTTS provider (cached)")
            return cls._coqui_instance
        
        # Unknown provider, use mock
        logger.warning(f"⚠️  Unknown TTS_PROVIDER '{provider_name}', using MockTTS")
        return MockTTS()

