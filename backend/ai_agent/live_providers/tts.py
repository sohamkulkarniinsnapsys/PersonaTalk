"""
Real Text-To-Speech Providers

This module implements actual text-to-speech synthesis.
Primary: Sarvam AI (cloud-based, supports 11 Indian languages)
Alternative: Coqui XTTS v2 (local, supports voice cloning)
"""

import os
import logging
import io
import asyncio
import requests
import uuid
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator

logger = logging.getLogger(__name__)


class SarvamTTS:
    """
    Real text-to-speech using Sarvam AI's TTS API.
    
    Sarvam AI provides multilingual text-to-speech for 11 Indian languages and English.
    
    Supported Languages:
    - Hindi (hi-IN)
    - Tamil (ta-IN)
    - Telugu (te-IN)
    - Kannada (kn-IN)
    - Malayalam (ml-IN)
    - Bengali (bn-IN)
    - Gujarati (gu-IN)
    - Marathi (mr-IN)
    - Punjabi (pa-IN)
    - Odia (od-IN)
    - English (en-IN) - Indian English
    
    Supported Voices (7 total):
    - Female: Anushka, Manisha, Vidya, Arya
    - Male: Abhilash, Karun, Hitesh
    
    Features:
    - Real-time streaming via WebSocket for ultra-low latency (~1s)
    - Pitch control: -0.75 to +0.75 semitones
    - Speed control: 0.5x to 2.0x
    - Loudness control: 0.3x to 3.0x
    
    Requires:
    - SARVAM_API_KEY environment variable
    - SARVAM_TTS_API_URL environment variable (optional, defaults to official API)
    """
    
    # Voice options available in Sarvam TTS
    AVAILABLE_VOICES = {
        # Female voices
        'anushka': {'gender': 'female', 'language_family': 'Indo-Aryan'},
        'manisha': {'gender': 'female', 'language_family': 'Dravidian'},
        'vidya': {'gender': 'female', 'language_family': 'Dravidian'},
        'arya': {'gender': 'female', 'language_family': 'Indo-Aryan'},
        # Male voices
        'abhilash': {'gender': 'male', 'language_family': 'Dravidian'},
        'karun': {'gender': 'male', 'language_family': 'Indo-Aryan'},
        'hitesh': {'gender': 'male', 'language_family': 'Indo-Aryan'},
    }
    
    # Language code mapping
    SUPPORTED_LANGUAGES = {
        'hi-IN': 'Hindi',
        'ta-IN': 'Tamil',
        'te-IN': 'Telugu',
        'kn-IN': 'Kannada',
        'ml-IN': 'Malayalam',
        'bn-IN': 'Bengali',
        'gu-IN': 'Gujarati',
        'mr-IN': 'Marathi',
        'pa-IN': 'Punjabi',
        'od-IN': 'Odia',
        'en-IN': 'English (Indian)',
    }
    
    def __init__(self):
        self.api_key = os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY required for SarvamTTS")
        
        # Sarvam AI TTS API endpoint
        self.api_url = os.environ.get("SARVAM_TTS_API_URL", "https://api.sarvam.ai/text-to-speech")
        
        # Normalize and validate API URL
        if not self.api_url.startswith('http'):
            self.api_url = "https://api.sarvam.ai/text-to-speech"
        
        # Default language (can be overridden per request based on voice config)
        self.default_language = "en-IN"
        
        # Default voice
        self.default_voice = "anushka"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "API-Subscription-Key": self.api_key,
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        })
        
        logger.info("✅ Initialized SarvamTTS with Sarvam AI API")
        logger.info(f"   Supported languages: {', '.join(self.SUPPORTED_LANGUAGES.values())}")
        logger.info(f"   Available voices: {', '.join(self.AVAILABLE_VOICES.keys())}")
    
    def _get_voice_and_language(self, voice_config: Dict[str, Any]) -> tuple[str, str]:
        """
        Determine voice and language from voice_config.
        
        voice_config can contain:
        - 'voice_id' or 'preset_id': Voice name (anushka, manisha, etc.)
        - 'language': Language code (en-IN, hi-IN, ta-IN, etc.)
        - 'language_code': Alternative key for language
        
        Returns:
            (voice_name: str, language_code: str)
        """
        # Get voice from config
        voice = (
            voice_config.get('voice_id') or 
            voice_config.get('preset_id') or 
            voice_config.get('voice') or
            self.default_voice
        )
        
        # Normalize voice name to lowercase
        voice = str(voice).lower().strip()
        
        # Validate voice exists
        if voice not in self.AVAILABLE_VOICES:
            logger.warning(f"⚠️  Voice '{voice}' not available; using default '{self.default_voice}'")
            voice = self.default_voice
        
        # Get language from config
        language_raw = (
            voice_config.get('language_code') or 
            voice_config.get('language') or
            voice_config.get('lang') or
            self.default_language
        )
        # Canonicalize to xx-XX (e.g., en-IN)
        language_raw = str(language_raw).strip().replace('_', '-')
        parts = language_raw.split('-')
        if len(parts) == 2:
            language = f"{parts[0].lower()}-{parts[1].upper()}"
        else:
            language = self.default_language
        
        # Validate language is supported
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"⚠️  Language '{language_raw}' not supported; using default '{self.default_language}'")
            logger.info(f"   Supported: {', '.join(self.SUPPORTED_LANGUAGES.keys())}")
            language = self.default_language
        
        return voice, language
    
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """
        Synthesize text to audio bytes using Sarvam AI API.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration dict containing:
                - voice_id: Voice name (anushka, manisha, vidya, arya, abhilash, karun, hitesh)
                - language_code: Language code (en-IN, hi-IN, ta-IN, etc.)
                - speed: Speech speed (0.5-2.0x, default 1.0)
                - pitch: Pitch adjustment (-0.75 to +0.75 semitones, default 0.0)
                
        Returns:
            Audio bytes (WAV format, 48kHz mono 16-bit PCM)
        """
        try:
            voice, language = self._get_voice_and_language(voice_config)
            
            # Get speech parameters with defaults
            speed = float(voice_config.get('speed', 1.0))
            pitch = float(voice_config.get('pitch', 0.0))
            
            # Validate ranges
            speed = max(0.5, min(2.0, speed))  # Clamp to 0.5-2.0x
            pitch = max(-0.75, min(0.75, pitch))  # Clamp to -0.75 to +0.75 semitones
            
            preview = (text[:80] + '...') if len(text) > 80 else text
            logger.info(f"🎙️  Starting TTS synthesis: {len(text)} chars, text='{preview}'")
            logger.info(f"   Voice: {voice} | Language: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
            logger.info(f"   Speed: {speed:.1f}x | Pitch: {pitch:+.2f}st")
            
            synthesis_start = time.time()
            
            # Prepare request payload
            payload = {
                "inputs": [text],
                "target_language_code": language,
                "speaker": voice,
                "pitch": pitch,
                "pace": speed,  # Sarvam uses 'pace' for speed control
                "loudness": 1.0,  # Default loudness
            }
            
            # Add unique request ID for tracking
            request_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
            logger.info(f"   Request ID: {request_id}")
            
            request_headers = {
                'X-Request-ID': request_id,
                'Content-Type': 'application/json',
            }
            
            # Call Sarvam TTS API
            retry_count = 0
            max_retries = 2
            backoff_delay = 0.5
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"🛰️  SarvamTTS POST {self.api_url}")
                    
                    resp = await asyncio.to_thread(
                        self.session.post,
                        self.api_url,
                        json=payload,
                        headers=request_headers,
                        timeout=30,
                    )
                    
                    logger.info(
                        "🛰️  SarvamTTS API response | status=%s | content_length=%s",
                        resp.status_code,
                        resp.headers.get('content-length', 'unknown'),
                    )
                    
                    if resp.ok:
                        break
                    elif retry_count < max_retries:
                        logger.warning(f"⚠️  HTTP {resp.status_code}, retrying in {backoff_delay}s...")
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2
                        retry_count += 1
                    else:
                        body_preview = resp.text[:500] if resp.text else "<no body>"
                        logger.error(f"❌ SarvamTTS API HTTP {resp.status_code}: {body_preview}")
                        resp.raise_for_status()
                
                except Exception as e:
                    if retry_count < max_retries:
                        logger.warning(f"⚠️  Connection failed: {e}, retrying in {backoff_delay}s...")
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2
                        retry_count += 1
                    else:
                        raise
            
            # Parse response - Sarvam returns audio bytes directly or in a JSON response
            audio_bytes = None
            
            # Check if response is binary audio or JSON
            content_type = (resp.headers.get('content-type') or '').lower()
            logger.info(f"   Content-Type: '{content_type}'")
            
            if ('audio' in content_type) or ('wav' in content_type) or ('mpeg' in content_type) or ('octet-stream' in content_type):
                # Direct audio bytes
                audio_bytes = resp.content
                logger.info(f"✅ Received direct audio: {len(audio_bytes)} bytes")
            elif 'json' in content_type or content_type == '':
                # JSON response containing audio
                try:
                    result = resp.json()
                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON in response")
                    raise

                # Attempt robust extraction of audio from JSON
                audio_bytes = self._extract_audio_from_json(result)

                # Check for error in response
                if not audio_bytes and isinstance(result, dict) and 'error' in result:
                    logger.error(f"❌ SarvamTTS API error: {result['error']}")
                    raise RuntimeError(f"TTS error: {result['error']}")

                if not audio_bytes:
                    # Log available keys for diagnostics
                    keys_preview = []
                    try:
                        if isinstance(result, dict):
                            keys_preview = list(result.keys())[:10]
                    except Exception:
                        pass
                    logger.error(f"❌ No audio found in JSON response. Keys: {keys_preview}")
                    raise RuntimeError("TTS JSON response did not contain audio")
            else:
                # Unknown/unsupported content type; do not assume binary audio
                logger.error(f"❌ Unsupported content type '{content_type}' for TTS response")
                raise RuntimeError(f"Unsupported content type: {content_type}")
            
            if not audio_bytes or len(audio_bytes) == 0:
                logger.error("❌ Synthesis failed: no audio received")
                raise RuntimeError("TTS synthesis returned empty audio")
            
            # Ensure audio is in correct format (WAV, 48kHz mono 16-bit)
            # Sarvam API should return 48kHz WAV, but verify
            audio_bytes = self._ensure_48k_mono_wav(audio_bytes)
            
            synthesis_time = time.time() - synthesis_start
            logger.info(f"✅ TTS synthesis complete: {synthesis_time:.2f}s total, {len(audio_bytes)} bytes")
            
            return audio_bytes
        
        except requests.exceptions.RequestException as e:
            body_preview = ""
            try:
                if hasattr(e, 'response') and e.response is not None and e.response.text:
                    body_preview = e.response.text[:500]
            except Exception:
                pass
            logger.error(f"❌ SarvamTTS API error: {e}{f' | body: {body_preview}' if body_preview else ''}")
            raise
        except Exception as e:
            logger.error(f"❌ SarvamTTS error: {e}", exc_info=True)
            raise
    
    async def stream(self, text: str, voice_config: Dict[str, Any]) -> AsyncIterator[bytes]:
        """
        Synthesize text and yield audio frames for streaming.
        
        Pre-buffers full synthesis, then yields 20ms chunks.
        """
        try:
            voice, language = self._get_voice_and_language(voice_config)
            
            logger.info(f"🎙️  Stream starting for text: {text[:80]}")
            
            # Full synthesis
            full_audio_pcm = await self.synthesize(text, voice_config)
            
            logger.info(f"✅ Stream synthesized, splitting into frames")
            
            # Convert to frames (20ms at 48kHz)
            frame_size = 960 * 2  # 20ms at 48kHz 16-bit mono = 960 samples * 2 bytes
            frame_count = 0
            
            for i in range(0, len(full_audio_pcm), frame_size):
                frame = full_audio_pcm[i:i+frame_size]
                if len(frame) > 0:
                    frame_count += 1
                    yield frame
                    await asyncio.sleep(0)  # Yield control to event loop
            
            logger.info(f"✅ Stream complete: yielded {frame_count} frames")
        
        except Exception as e:
            logger.error(f"❌ Stream synthesis failed: {e}")
            raise
    
    def _ensure_48k_mono_wav(self, audio_bytes: bytes) -> bytes:
        """
        Ensure audio is in correct format: WAV, 48kHz, mono, 16-bit PCM.
        Sarvam API should return this format, but verify and convert if needed.
        """
        try:
            import wave
            import io
            from pydub import AudioSegment
            
            # Try to parse as WAV
            try:
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                    sample_rate = wav.getframerate()
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    
                    logger.info(f"   Audio format: {sample_rate}Hz {channels}ch {sample_width*8}bit")
                    
                    # If already 48kHz mono 16-bit, return as-is
                    if sample_rate == 48000 and channels == 1 and sample_width == 2:
                        return audio_bytes
            except Exception:
                logger.debug("Could not parse as WAV, attempting conversion")
            
            # Convert using pydub if needed (auto-detect format)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Enforce mono, 48kHz, 16-bit
            if audio.frame_rate != 48000 or audio.channels != 1 or audio.sample_width != 2:
                logger.info(f"   Converting to 48kHz mono 16-bit...")
                audio = audio.set_channels(1).set_frame_rate(48000).set_sample_width(2)
            
            # Export to WAV bytes
            return audio.export(format="wav").read()
        
        except Exception as e:
            logger.warning(f"⚠️  Could not verify/convert audio format: {e}")
            # Return as-is, hope it's correct
            return audio_bytes

    def _extract_audio_from_json(self, obj: Any) -> bytes | None:
        """
        Recursively search JSON for audio content.
        Supports base64/data URI strings and audio URLs.
        """
        try:
            import base64
            import re
        except Exception:
            pass

        def try_decode_base64(s: str) -> bytes | None:
            try:
                if s.startswith('data:') and ',' in s:
                    _, b64 = s.split(',', 1)
                    return base64.b64decode(b64)
                # Heuristic: long, base64-like characters
                if len(s) > 1024 and re.fullmatch(r'[A-Za-z0-9+/=\n\r]+', s):
                    return base64.b64decode(s)
            except Exception:
                return None
            return None

        def is_probably_wav(b: bytes) -> bool:
            return isinstance(b, (bytes, bytearray)) and len(b) > 16 and b[:4] == b'RIFF'

        # String: base64 or URL
        if isinstance(obj, str):
            decoded = try_decode_base64(obj)
            if decoded:
                logger.info(f"✅ Decoded base64/data URI audio: {len(decoded)} bytes")
                return decoded
            if obj.startswith('http://') or obj.startswith('https://'):
                try:
                    r = self.session.get(obj, timeout=30)
                    ct = (r.headers.get('content-type') or '').lower()
                    if r.ok and ('audio' in ct or 'wav' in ct or 'octet-stream' in ct):
                        logger.info(f"✅ Fetched audio URL: {len(r.content)} bytes")
                        return r.content
                except Exception as e:
                    logger.warning(f"⚠️  Failed to fetch audio URL: {e}")
            return None

        # Bytes: quick RIFF check
        if isinstance(obj, (bytes, bytearray)):
            return obj if is_probably_wav(obj) else None

        # Dict: check common keys first, then deep search
        if isinstance(obj, dict):
            preferred = ('output','audio','audio_wav','audio_base64','wav','data','result','outputs','audio_url')
            for k in preferred:
                if k in obj:
                    found = self._extract_audio_from_json(obj[k])
                    if found:
                        return found
            for k, v in obj.items():
                if 'audio' in k.lower() or 'wav' in k.lower():
                    found = self._extract_audio_from_json(v)
                    if found:
                        return found
            for v in obj.values():
                found = self._extract_audio_from_json(v)
                if found:
                    return found
            return None

        # List: check items
        if isinstance(obj, list):
            for item in obj:
                found = self._extract_audio_from_json(item)
                if found:
                    return found
            return None

        return None
