"""
Real Speech-To-Text Providers

This module implements actual speech-to-text recognition.
Primary: Local Whisper (free, offline, no API required)
Fallback: Google Speech-to-Text via speech_recognition
"""

import os
import logging
import io
import asyncio
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class WhisperLocalSTT:
    """
    Local speech-to-text using OpenAI's Whisper model (runs on your GPU/CPU).
    
    Advantages:
    - FREE (no API costs)
    - OFFLINE (no internet required)
    - FAST (GPU accelerated)
    - RELIABLE (no network failures)
    - ACCURATE (95%+ accuracy on clear speech)
    
    Requires:
    - pip install openai-whisper
    - Automatically downloads model on first use (~2.9 GB for base model)
    """
    
    def __init__(self):
        try:
            import whisper
            
            # Load model - use 'base' for good balance of accuracy and speed
            # Options: tiny, base, small, medium, large
            # 'base' is more reliable for conversational speech (less prone to hallucinations)
            model_name = os.environ.get("WHISPER_MODEL", "base")
            logger.info(f"🔄 Loading Whisper model: {model_name} (this may take 30s on first run)...")
            
            # Use GPU if available, otherwise CPU
            device = "cuda" if self._has_gpu() else "cpu"
            self.model = whisper.load_model(model_name, device=device)
            
            logger.info(f"✅ Initialized WhisperLocalSTT with model={model_name} on device={device}")
            logger.info(f"ℹ️  Using '{model_name}' model for reliable conversational speech recognition")
        except ImportError:
            logger.error(
                "❌ openai-whisper not installed. Install with: pip install openai-whisper"
            )
            raise RuntimeError(
                "whisper package required for WhisperLocalSTT. Install with: pip install openai-whisper"
            )
    
    def _has_gpu(self) -> bool:
        """Check if CUDA/GPU is available."""
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            logger.info(f"🎮 GPU available: {has_gpu}")
            return has_gpu
        except Exception:
            logger.info("🎮 GPU not available, will use CPU")
            return False
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text using local Whisper model.
        
        Args:
            audio_bytes: Raw audio data (PCM s16, 48kHz mono)
            
        Returns:
            Transcribed text
        """
        try:
            import io
            from pydub import AudioSegment
            import numpy as np
            import tempfile
            
            logger.info(f"🎤 WhisperLocalSTT: Transcribing {len(audio_bytes)} bytes of audio...")
            logger.info(f"   Audio info: {len(audio_bytes)} bytes = {len(audio_bytes) / (48000 * 2):.2f}s at 48kHz s16")
            
            # Convert PCM to WAV in memory
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate=48000, num_channels=1, sample_width=2)
            logger.info(f"   WAV conversion: PCM {len(audio_bytes)} → WAV {len(wav_bytes)} bytes")
            
            # Load audio with pydub
            audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            logger.info(f"   Audio loaded: {audio.frame_rate}Hz, {audio.channels} channels")
            
            # Convert to mono 16kHz for Whisper (Whisper requirement)
            audio = audio.set_channels(1).set_frame_rate(16000)
            logger.info(f"   Resampled to 16kHz, {audio.channels} channel(s)")

            # Normalize audio: apply gentle RMS-based gain to ensure consistent volume
            # This helps with quiet speech and prevents clipping
            target_rms = -20.0  # Target RMS in dBFS (good for speech)
            current_rms = audio.dBFS
            if current_rms < -40.0:  # Very quiet audio
                gain_needed = target_rms - current_rms
                audio = audio.apply_gain(min(gain_needed, 12.0))  # Cap gain at +12dB
                logger.info(f"   Applied gain: +{min(gain_needed, 12.0):.1f} dB (quiet audio boost)")
            
            # Prevent clipping: if peak is above -1 dBFS, reduce gain
            if audio.max_dBFS is not None and audio.max_dBFS > -1.0:
                gain = -1.0 - audio.max_dBFS
                audio = audio.apply_gain(gain)
                logger.info(f"   Applied headroom gain: {gain:.2f} dB to avoid clipping")
            
            # Convert to numpy array - keep as int16 first
            samples_int16 = np.array(audio.get_array_of_samples(), dtype=np.int16)
            logger.info(f"   Samples shape: {samples_int16.shape}, dtype: {samples_int16.dtype}")
            logger.info(f"   Sample range: [{samples_int16.min()}, {samples_int16.max()}]")
            
            # Normalize to [-1, 1] range - use 32767 for proper s16 normalization
            samples = samples_int16.astype(np.float32) / 32767.0
            logger.info(f"   Normalized range: [{samples.min():.3f}, {samples.max():.3f}]")
            
            # Whisper expects audio as a file path or AudioSegment, NOT numpy array
            # Save to temporary WAV file and let Whisper load it
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                audio.export(tmp_path, format="wav")
            
            logger.info(f"   Loading into Whisper from temp file...")
            
            try:
                # Run transcription on local model with optimized parameters
                result = self.model.transcribe(
                    tmp_path,
                    language="en",
                    fp16=self._has_gpu(),  # Use fp16 if GPU available
                    verbose=False,
                    temperature=0.0,       # Fully deterministic decode
                    beam_size=5,           # Balanced beam for conversational speech (not too high)
                    best_of=5,
                    compression_ratio_threshold=2.4,  # Slightly lower to catch repetitive hallucinations
                    logprob_threshold=-1.0,
                    condition_on_previous_text=False,  # Avoid context bleed between utterances
                    without_timestamps=True,
                    initial_prompt=(
                        "Transcribe natural conversational speech about software and systems work. "
                        "Common phrases: 'I am working on backend systems', 'request latency', 'database performance', "
                        "'API design', 'frontend issues', 'fixing bugs', 'building features'. "
                        "Technical terms: JavaScript, Python, React, Django, PostgreSQL, Redis, AWS, Docker, Kubernetes. "
                        "Transcribe exactly what is said without adding extra words."
                    ),
                )
                
                transcript = result.get("text", "").strip()
                logger.info(f"   Whisper result keys: {list(result.keys())}")
                logger.info(f"   Full result: {result}")
                
                # Enforce confidence gating: check avg_logprob from first segment
                # Relaxed threshold: -1.5 instead of -1.0 (conversational speech has higher variance)
                segments = result.get("segments") or []
                if segments:
                    seg0 = segments[0]
                    avg_logprob = seg0.get("avg_logprob")
                    no_speech_prob = seg0.get("no_speech_prob", 0.0)
                    logger.info(f"   Segment[0] avg_logprob={avg_logprob}, no_speech_prob={no_speech_prob}")
                    
                    # Only reject if VERY low confidence (< -1.5) to avoid false rejections
                    # Real conversational speech typically has avg_logprob between -0.5 and -1.2
                    if avg_logprob is not None and avg_logprob < -1.5:
                        logger.warning(f"⚠️  Very low confidence (avg_logprob={avg_logprob:.2f} < -1.5); asking user to repeat")
                        return "I didn't catch that, could you please repeat?"
                    
                    # Only reject if EXTREMELY high no-speech (> 0.85) - the 0.6 threshold was rejecting valid speech
                    if no_speech_prob > 0.85:
                        logger.warning(f"⚠️  Very high no-speech probability ({no_speech_prob:.2f}); likely silence")
                        return "I didn't catch that, could you please repeat?"
                
                if transcript:
                    # Additional validation: reject suspiciously short transcripts that might be artifacts
                    if len(transcript) < 2 and transcript.lower() in ['a', 'i', 'the', 'to', 'of']:
                        logger.warning(f"⚠️  Single-letter artifact detected: '{transcript}'; treating as noise")
                        return "I didn't catch that, could you please repeat?"
                    
                    logger.info(f"✅ Local Whisper transcribed: '{transcript}'")
                    return transcript
                else:
                    logger.warning("⚠️  Local Whisper returned empty transcript")
                    return "I didn't catch that"
            finally:
                # Clean up temp file
                import os as os_module
                try:
                    os_module.remove(tmp_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ WhisperLocalSTT error: {e}", exc_info=True)
            return "I couldn't understand that"
    
    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 48000, num_channels: int = 1, sample_width: int = 2) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        import struct
        
        channels = num_channels
        byte_rate = sample_rate * channels * sample_width
        block_align = channels * sample_width
        
        wav_header = io.BytesIO()
        
        # RIFF chunk
        wav_header.write(b'RIFF')
        wav_header.write(struct.pack('<I', 36 + len(pcm_bytes)))
        wav_header.write(b'WAVE')
        
        # fmt sub-chunk
        wav_header.write(b'fmt ')
        wav_header.write(struct.pack('<I', 16))
        wav_header.write(struct.pack('<H', 1))
        wav_header.write(struct.pack('<H', channels))
        wav_header.write(struct.pack('<I', sample_rate))
        wav_header.write(struct.pack('<I', byte_rate))
        wav_header.write(struct.pack('<H', block_align))
        wav_header.write(struct.pack('<H', sample_width * 8))
        
        # data sub-chunk
        wav_header.write(b'data')
        wav_header.write(struct.pack('<I', len(pcm_bytes)))
        
        return wav_header.getvalue() + pcm_bytes


class GroqWhisperSTT:
    """
    Real speech-to-text using Groq's Whisper API.
    
    Groq provides fast Whisper inference - this is production-ready STT.
    
    Requires:
    - GROQ_API_KEY environment variable
    - Audio in WAV or other format that Whisper accepts
    """
    
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY required for GroqWhisperSTT")
        
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self.model = "whisper-large-v3"  # Track which model we're using for logging
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })
        
        logger.info(f"✅ Initialized GroqWhisperSTT with Groq API (model={self.model})")
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text using Groq Whisper API.
        
        Args:
            audio_bytes: Raw audio data (PCM s16, 48kHz mono)
            
        Returns:
            Transcribed text
        """
        try:
            # Convert raw PCM to WAV format for Whisper
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate=48000, num_channels=1, sample_width=2)
            
            # Prepare file for upload
            files = {
                'file': ('audio.wav', io.BytesIO(wav_bytes), 'audio/wav'),
                # Use Groq Whisper large model (current production audio model)
                'model': (None, 'whisper-large-v3'),
                'language': (None, 'en'),
            }
            
            logger.info(f"🎤 GroqWhisperSTT: Sending {len(audio_bytes)} bytes for transcription...")
            
            # Call Groq Whisper API
            response = self.session.post(self.api_url, files=files, timeout=30)
            logger.info(
                "🛰️ GroqWhisperSTT POST https://api.groq.com/openai/v1/audio/transcriptions | model=%s | wav_bytes=%s | status=%s",
                self.model,
                len(wav_bytes),
                response.status_code,
            )

            # If Groq returns non-2xx, capture body for debugging before raising
            if not response.ok:
                body_preview = response.text[:500] if response.text else "<no body>"
                logger.error(
                    "❌ Groq Whisper HTTP %s: %s",
                    response.status_code,
                    body_preview,
                )
                response.raise_for_status()

            result = response.json()
            transcript = result.get('text', '').strip()
            
            if transcript:
                logger.info(f"✅ Groq Whisper transcribed: '{transcript}'")
                # Validate transcription quality: flag if suspiciously short
                if len(transcript) < 2:
                    logger.warning(f"⚠️  WARNING - Very short Whisper output ({len(transcript)} chars): '{transcript}' - may indicate audio quality issues")
                return transcript
            else:
                logger.warning("⚠️  Groq Whisper returned empty transcript")
                return "I didn't catch that"
                
        except requests.exceptions.RequestException as e:
            body_preview = ""
            try:
                if e.response is not None and e.response.text:
                    body_preview = e.response.text[:500]
            except Exception:
                pass
            logger.error(
                "❌ Groq Whisper API error: %s%s",
                e,
                f" | body: {body_preview}" if body_preview else "",
            )
            return "Sorry, there was an error processing your speech"
        except Exception as e:
            logger.error(f"❌ GroqWhisperSTT error: {e}")
            return "I couldn't understand that"
    
    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 48000, num_channels: int = 1, sample_width: int = 2) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        import struct
        
        # WAV file header
        channels = num_channels
        sample_rate = sample_rate
        byte_rate = sample_rate * channels * sample_width
        block_align = channels * sample_width
        
        # Create WAV header
        wav_header = io.BytesIO()
        
        # RIFF chunk
        wav_header.write(b'RIFF')
        wav_header.write(struct.pack('<I', 36 + len(pcm_bytes)))
        wav_header.write(b'WAVE')
        
        # fmt sub-chunk
        wav_header.write(b'fmt ')
        wav_header.write(struct.pack('<I', 16))  # fmt chunk size
        wav_header.write(struct.pack('<H', 1))   # PCM format
        wav_header.write(struct.pack('<H', channels))
        wav_header.write(struct.pack('<I', sample_rate))
        wav_header.write(struct.pack('<I', byte_rate))
        wav_header.write(struct.pack('<H', block_align))
        wav_header.write(struct.pack('<H', sample_width * 8))  # bits per sample
        
        # data sub-chunk
        wav_header.write(b'data')
        wav_header.write(struct.pack('<I', len(pcm_bytes)))
        
        return wav_header.getvalue() + pcm_bytes


class SarvamSTT:
    """
    Real speech-to-text using Sarvam AI's STT API.
    
    Sarvam AI provides multilingual speech-to-text for Indian languages and English.
    
    Requires:
    - SARVAM_API_KEY environment variable
    - Audio in WAV format
    """
    
    def __init__(self):
        self.api_key = os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY required for SarvamSTT")
        
        # Sarvam AI API endpoints
        self.api_url = os.environ.get("SARVAM_API_URL", "https://api.sarvam.ai/speech-to-text")

        # Normalize API URL robustly (case-insensitive, fixes typos like speech-to-textt, forces path)
        def _normalize_url(url: str) -> str:
            from urllib.parse import urlsplit, urlunsplit

            if not url:
                return "https://api.sarvam.ai/speech-to-text"

            url = url.strip()
            parts = urlsplit(url)

            scheme = parts.scheme or "https"
            netloc = parts.netloc or parts.path.split("/")[0]
            path = parts.path if parts.netloc else "/" + "/".join(parts.path.split("/")[1:])

            # Normalize the path to the expected endpoint regardless of typos/case
            lower_path = (path or "").lower()
            if "speech-to-text" not in lower_path:
                path = "/speech-to-text"
            else:
                # Fix doubled letters and ensure exact path
                path = "/speech-to-text"

            normalized = urlunsplit((scheme, netloc, path, "", ""))
            return normalized

        normalized_api_url = _normalize_url(self.api_url)
        if normalized_api_url != self.api_url:
            logger.warning("⚠️  Normalized SARVAM_API_URL from '%s' to '%s'", self.api_url, normalized_api_url)
            self.api_url = normalized_api_url

        # Docs: Saarika supports 11 Indic languages incl. en-IN; use latest model by default
        self.model = os.environ.get("SARVAM_MODEL", "saarika:v2.5")  # valid: saarika:v1|v2|v2.5|flash
        configured_language = os.environ.get("SARVAM_LANGUAGE_CODE", "en-IN") or "en-IN"
        if configured_language.lower() in {"auto", "unknown", ""}:
            configured_language = "unknown"

        # Sarvam API accepts en-IN (not plain en); map common inputs to supported code
        if configured_language.lower() in {"en", "en-in", "english"}:
            configured_language = "en-IN"

        self.language_code = configured_language
        supported_languages = {
            "en-IN",
            "hi-IN",
            "bn-IN",
            "ta-IN",
            "te-IN",
            "gu-IN",
            "kn-IN",
            "ml-IN",
            "mr-IN",
            "pa-IN",
            "od-IN",
            "unknown",  # auto-detect
        }
        if self.language_code not in supported_languages:
            logger.warning("⚠️  SARVAM_LANGUAGE_CODE '%s' is not in the documented set; requests may fail", self.language_code)
        self.session = requests.Session()
        self.session.headers.update({
            "API-Subscription-Key": self.api_key,
        })
        
        logger.info("✅ Initialized SarvamSTT with Sarvam AI API")
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text using Sarvam AI API.
        
        Sarvam Saarika:v2.5 is optimized for:
        - 16kHz sample rate (not 48kHz) - resampling improves accuracy per docs
        - PCM s16le format with explicit codec declaration
        - Indian English (en-IN) with telephony optimization
        
        Args:
            audio_bytes: Raw audio data (PCM s16, 48kHz mono from WebRTC)
            
        Returns:
            Transcribed text
        """
        try:
            # CRITICAL: Sarvam docs state "API works best with audio files sampled at 16kHz"
            # Resample 48kHz -> 16kHz to match model optimization and improve accuracy
            resample_to_16k = os.environ.get("SARVAM_RESAMPLE_16K", "True").lower() == "true"
            
            if resample_to_16k:
                try:
                    from pydub import AudioSegment
                    
                    # Convert PCM 48kHz to WAV
                    wav_48k = self._pcm_to_wav(audio_bytes, sample_rate=48000, num_channels=1, sample_width=2)
                    audio = AudioSegment.from_wav(io.BytesIO(wav_48k))
                    
                    # Enforce mono, 16kHz, 16-bit
                    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

                    # CRITICAL: Minimal, surgical normalization to preserve speech fidelity
                    # Over-processing (aggressive gain, filters) distorts consonants and confuses STT
                    # Example: Backend vs Background confusion was caused by +12dB gain clipping fricatives
                    target_rms = -20.0  # Sarvam's sweet spot for conversational speech
                    current_rms = audio.dBFS
                    if current_rms is None or current_rms == float("-inf"):
                        current_rms = -60.0
                    gain_needed = target_rms - current_rms
                    if gain_needed > 0.5:  # Only apply if meaningful
                        # REDUCED MAX: +3dB instead of +12dB to avoid distortion
                        # Speech is clearer with slight underamplification than over-amplification
                        audio = audio.apply_gain(min(gain_needed, 3.0))
                        logger.info(f"   Applied gentle gain: +{min(gain_needed, 3.0):.1f}dB")
                    
                    # Safety limiter at -1dB (aggressive headroom to avoid clipping)
                    if audio.max_dBFS is not None and audio.max_dBFS > -1.0:
                        headroom = -1.0 - audio.max_dBFS
                        audio = audio.apply_gain(headroom)
                        logger.info(f"   Applied headroom: {headroom:.2f}dB")
                    
                    # SKIP high-pass filter entirely
                    # 80 Hz filter removes important fricative/consonant content (f, v, s, sh, th sounds)
                    # These are critical for distinguishing: backend vs background, web vs weird, etc.
                    # Sarvam's model is robust to low-freq rumble; prefer clarity over cleanup
                    # If rumble is excessive, Sarvam will flag low confidence and we'll retry

                    wav_bytes = audio.export(format="wav").read()
                    
                    logger.info(f"🎤 SarvamSTT: Resampled 48kHz ({len(audio_bytes)} bytes) → 16kHz ({len(wav_bytes)} WAV bytes), mono s16, surgically normalized (target=-20dBFS, max_gain=+3dB, no HPF)")
                except Exception as e:
                    logger.warning(f"⚠️  Resampling/normalization to 16kHz failed ({e}); using original 48kHz")
                    wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate=48000, num_channels=1, sample_width=2)
            else:
                wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate=48000, num_channels=1, sample_width=2)
            
            # Prepare multipart form data (use BytesIO variable for reuse in retry)
            audio_stream = io.BytesIO(wav_bytes)
            files = {
                'file': ('audio.wav', audio_stream, 'audio/wav'),
            }
            
            input_codec = os.environ.get("SARVAM_INPUT_CODEC", "pcm_s16le")
            if input_codec and input_codec.lower() in ["pcm_s16le", "pcm_l16", "pcm_raw"]:
                logger.info(f"   Using input_audio_codec={input_codec}")
            else:
                input_codec = None

            async def call_sarvam(lang_code: str):
                # Prepare form data
                data = {
                    'language_code': lang_code,
                    'model': self.model,
                }
                if input_codec:
                    data['input_audio_codec'] = input_codec

                retry_count = 0
                max_retries = 2
                backoff_delay = 0.5
                while retry_count <= max_retries:
                    try:
                        resp = self.session.post(
                            self.api_url,
                            files=files,
                            data=data,
                            timeout=30
                        )
                        logger.info(
                            "🛰️ SarvamSTT POST %s | model=%s | lang=%s | codec=%s | wav_bytes=%s | status=%s",
                            self.api_url,
                            self.model,
                            lang_code,
                            input_codec,
                            len(wav_bytes),
                            resp.status_code,
                        )
                        return resp
                    except Exception as e:
                        if retry_count < max_retries:
                            logger.warning(f"⚠️  Sarvam API connection failed: {e}, retrying in {backoff_delay}s...")
                            await asyncio.sleep(backoff_delay)
                            backoff_delay *= 2
                            retry_count += 1
                            audio_stream.seek(0)
                        else:
                            raise

            # First attempt with configured language
            response = await call_sarvam(self.language_code)
            
            # Handle errors
            if not response.ok:
                body_preview = response.text[:500] if response.text else "<no body>"
                logger.error(
                    "❌ Sarvam AI HTTP %s: %s",
                    response.status_code,
                    body_preview,
                )
                response.raise_for_status()
            
            result = response.json()

            def extract_transcript(res_json):
                return (res_json.get('transcript') or '').strip(), res_json.get('confidence')

            transcript, confidence = extract_transcript(result)

            # If low confidence or empty, try a fallback language (unknown/auto then en-IN)
            fallback_langs = []
            if not transcript or (confidence is not None and confidence < 0.6):
                if self.language_code != 'unknown':
                    fallback_langs.append('unknown')
                if self.language_code != 'en-IN':
                    fallback_langs.append('en-IN')

            for fallback_lang in fallback_langs:
                try:
                    audio_stream.seek(0)
                    resp_fb = await call_sarvam(fallback_lang)
                    if not resp_fb or not resp_fb.ok:
                        continue
                    fb_json = resp_fb.json()
                    fb_transcript, fb_conf = extract_transcript(fb_json)
                    logger.info(
                        "🛰️ SarvamSTT fallback lang=%s transcript='%s' conf=%s",
                        fallback_lang,
                        fb_transcript,
                        f"{fb_conf:.2f}" if fb_conf is not None else "n/a",
                    )
                    if fb_transcript:
                        transcript, confidence = fb_transcript, fb_conf
                        break
                except Exception as retry_err:
                    logger.warning(f"⚠️  Sarvam AI fallback ({fallback_lang}) failed: {retry_err}")

            if transcript:
                logger.info(f"✅ Sarvam AI transcribed: '{transcript}'")
                if confidence is not None:
                    logger.info(f"   Confidence: {confidence:.2f}")
                    if confidence < 0.5:
                        logger.warning(f"⚠️  Low confidence ({confidence:.2f} < 0.5 threshold); asking user to repeat")
                        return "I didn't catch that, could you please repeat?"
                return self._postprocess_transcript(transcript)

            logger.warning("⚠️  Sarvam AI returned empty transcript")
            if len(wav_bytes) < 100:
                logger.error(f"❌ Audio size too small ({len(wav_bytes)} bytes) - VAD likely dropped utterance")
                logger.info(f"   This usually means: user spoke too quietly, or VAD threshold too high")
                return "Your speech was too quiet. Please speak up."
            return "I didn't catch that, could you please repeat?"
                
        except requests.exceptions.RequestException as e:
            body_preview = ""
            try:
                if e.response is not None and e.response.text:
                    body_preview = e.response.text[:500]
            except Exception:
                pass
            logger.error(
                "❌ Sarvam AI API error: %s%s",
                e,
                f" | body: {body_preview}" if body_preview else "",
            )
            return "Sorry, there was an error processing your speech"
        except Exception as e:
            logger.error(f"❌ SarvamSTT error: {e}", exc_info=True)
            return "I couldn't understand that"
    
    def _postprocess_transcript(self, text: str) -> str:
        """Apply small, safe corrections for common STT misrecognitions.
        
        These are phonetically similar but wrong in context.
        Regex-based, domain-aware post-processing as a last-resort correction.
        """
        try:
            import re
            original = text
            t = text
            
            # CRITICAL: Background vs Backend confusion fix
            # STT confuses these when audio clarity is compromised (e.g., from aggressive normalization)
            # Heuristic: In a tech interview, "backend" is vastly more common than "background systems"
            t = re.sub(r"\bbackground systems\b", "backend systems", t, flags=re.IGNORECASE)
            t = re.sub(r"\bbackground system\b", "backend system", t, flags=re.IGNORECASE)
            # Single word context: "backend " vs "background " in tech phrases
            t = re.sub(r"\bbackground development\b", "backend development", t, flags=re.IGNORECASE)
            t = re.sub(r"\bbackground services\b", "backend services", t, flags=re.IGNORECASE)
            
            # "weird application" -> "web application"
            t = re.sub(r"\bweird application\b", "web application", t, flags=re.IGNORECASE)
            # Horizontal/vertical "light" -> "scaling" when paired
            t = re.sub(r"\bhorizontal and vertical light\b", "horizontal and vertical scaling", t, flags=re.IGNORECASE)
            t = re.sub(r"\bvertical light\b", "vertical scaling", t, flags=re.IGNORECASE)
            t = re.sub(r"\bhorizontal light\b", "horizontal scaling", t, flags=re.IGNORECASE)
            # Horizontal/vertical "skimming" -> "scaling" (pricing vs system design confusion)
            t = re.sub(r"\bhorizontal and vertical skimming\b", "horizontal and vertical scaling", t, flags=re.IGNORECASE)
            t = re.sub(r"\bvertical skimming\b", "vertical scaling", t, flags=re.IGNORECASE)
            t = re.sub(r"\bhorizontal skimming\b", "horizontal scaling", t, flags=re.IGNORECASE)
            # "space" -> "scaling" in system design context
            t = re.sub(r"\bhorizontal and vertical space\b", "horizontal and vertical scaling", t, flags=re.IGNORECASE)
            if t != original:
                logger.info(f"🔧 Post-processed transcript: '{original}' → '{t}'")
            return t
        except Exception:
            return text
    
    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 48000, num_channels: int = 1, sample_width: int = 2) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        import struct
        
        channels = num_channels
        byte_rate = sample_rate * channels * sample_width
        block_align = channels * sample_width
        
        wav_header = io.BytesIO()
        
        # RIFF chunk
        wav_header.write(b'RIFF')
        wav_header.write(struct.pack('<I', 36 + len(pcm_bytes)))
        wav_header.write(b'WAVE')
        
        # fmt sub-chunk
        wav_header.write(b'fmt ')
        wav_header.write(struct.pack('<I', 16))
        wav_header.write(struct.pack('<H', 1))
        wav_header.write(struct.pack('<H', channels))
        wav_header.write(struct.pack('<I', sample_rate))
        wav_header.write(struct.pack('<I', byte_rate))
        wav_header.write(struct.pack('<H', block_align))
        wav_header.write(struct.pack('<H', sample_width * 8))
        
        # data sub-chunk
        wav_header.write(b'data')
        wav_header.write(struct.pack('<I', len(pcm_bytes)))
        
        return wav_header.getvalue() + pcm_bytes


class SpeechRecognitionSTT:
    """
    Fallback STT using local speech_recognition library with Google Speech-to-Text.
    
    Requires:
    - pip install SpeechRecognition pydub
    - Internet connection
    
    This is slower than Groq but works offline with local models too.
    """
    
    def __init__(self):
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            logger.info("✅ Initialized SpeechRecognitionSTT")
        except ImportError:
            logger.error("❌ speech_recognition not installed. Install with: pip install SpeechRecognition pydub")
            raise RuntimeError("speech_recognition package required for SpeechRecognitionSTT")
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio using Google Speech-to-Text (via speech_recognition).
        
        Args:
            audio_bytes: Raw audio data (PCM s16, 48kHz mono)
            
        Returns:
            Transcribed text
        """
        try:
            import speech_recognition as sr
            import io
            from pydub import AudioSegment
            
            logger.info(f"🎤 SpeechRecognitionSTT: Processing {len(audio_bytes)} bytes...")
            
            # Convert PCM to WAV in memory
            wav_bytes = GroqWhisperSTT._pcm_to_wav(audio_bytes, sample_rate=48000, num_channels=1, sample_width=2)
            
            # Load with pydub
            audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            
            # Convert to audio_data for recognizer
            audio_data = sr.AudioData(audio.raw_data, audio.frame_rate, audio.sample_width)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio_data)
            logger.info(f"✅ Speech-to-Text transcribed: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"❌ Speech-to-Text error: {e}")
            return "I couldn't understand that"

