"""
Silero VAD integration for reliable speech detection.

This module wraps the Silero VAD model to provide frame-by-frame speech classification,
distinguishing real human speech from background noise, keyboard sounds, mic pops, etc.
"""

import logging
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD wrapper for real-time speech detection.
    
    Provides frame-by-frame classification of audio as "speech" or "non-speech"
    using the pre-trained Silero VAD model. This is significantly more reliable
    than simple energy/volume thresholds for distinguishing speech from noise.
    
    Usage:
        vad = SileroVAD()
        is_speech = vad.is_speech(audio_chunk_48khz_mono_s16)
    """
    
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        Initialize Silero VAD model.
        
        Args:
            threshold: Confidence threshold (0.0-1.0). Higher = stricter.
                      Default 0.5 is balanced. Use 0.6-0.7 for noisy environments.
            sample_rate: Model expects 16kHz. We'll resample internally if needed.
        """
        self.threshold = threshold
        self.model_sample_rate = sample_rate  # Silero expects 16kHz
        self.model = None
        self.utils = None
        
        # Lazy initialization - only load model when first used
        self._initialized = False
        
        logger.info(f"SileroVAD configured with threshold={threshold}, sample_rate={sample_rate}")
    
    def _initialize_model(self):
        """Lazy model loading to avoid startup delays."""
        if self._initialized:
            return
        
        try:
            # Load Silero VAD from torch hub
            # This downloads the model on first run (~1MB), then caches locally
            logger.info("Loading Silero VAD model from torch.hub...")
            
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False  # Use PyTorch model for better CPU performance
            )
            
            # Extract utility functions
            (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks) = self.utils
            
            self.get_speech_timestamps = get_speech_timestamps
            
            # Set model to eval mode for inference
            self.model.eval()
            
            self._initialized = True
            logger.info("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Silero VAD model: {e}", exc_info=True)
            logger.warning("⚠️  Falling back to energy-based detection")
            self._initialized = False
    
    def is_speech(self, audio_chunk: bytes, input_sample_rate: int = 48000) -> bool:
        """
        Classify audio chunk as speech or non-speech.
        
        Args:
            audio_chunk: Raw audio bytes (16-bit PCM, mono)
            input_sample_rate: Sample rate of input audio (default 48kHz from WebRTC)
        
        Returns:
            True if speech detected, False otherwise
        """
        # Ensure model is loaded
        if not self._initialized:
            self._initialize_model()
        
        # If model failed to load, fall back to False (don't block on errors)
        if not self._initialized or self.model is None:
            return False
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample from 48kHz to 16kHz if needed (Silero expects 16kHz)
            if input_sample_rate != self.model_sample_rate:
                audio_np = self._resample(audio_np, input_sample_rate, self.model_sample_rate)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_np)
            
            # Run inference (no gradient computation needed)
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.model_sample_rate).item()
            
            # Classify based on threshold
            is_speech = speech_prob >= self.threshold
            
            # Log periodically for debugging (every ~1 second of audio)
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 0
            
            if self._frame_count % 50 == 0:
                logger.debug(
                    f"Silero VAD: prob={speech_prob:.3f}, threshold={self.threshold:.3f}, "
                    f"is_speech={is_speech}"
                )
            
            return is_speech
            
        except Exception as e:
            logger.warning(f"Silero VAD inference error: {e}")
            return False  # Safe default: don't trigger on errors
    
    def get_speech_probability(self, audio_chunk: bytes, input_sample_rate: int = 48000) -> float:
        """
        Get raw speech probability without thresholding.
        
        Useful for debugging or dynamic threshold adjustment.
        
        Returns:
            Probability value 0.0-1.0, or 0.0 on error
        """
        if not self._initialized:
            self._initialize_model()
        
        if not self._initialized or self.model is None:
            return 0.0
        
        try:
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            if input_sample_rate != self.model_sample_rate:
                audio_np = self._resample(audio_np, input_sample_rate, self.model_sample_rate)
            
            audio_tensor = torch.from_numpy(audio_np)
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.model_sample_rate).item()
            
            return speech_prob
            
        except Exception:
            return 0.0
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Simple resampling using scipy for 48kHz → 16kHz conversion.
        
        This is a lightweight alternative to more complex resampling libraries.
        """
        try:
            from scipy import signal
            
            # Calculate resampling ratio
            ratio = target_sr / orig_sr
            
            # Use scipy's resample for quality
            num_samples = int(len(audio) * ratio)
            resampled = signal.resample(audio, num_samples)
            
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, returning original audio")
            return audio
    
    def reset_state(self):
        """
        Reset any internal state (for multi-frame processing).
        
        Call this between different audio streams or when starting new detection.
        """
        # Silero VAD is stateless for single-frame inference, but we reset counters
        self._frame_count = 0
        logger.debug("Silero VAD state reset")


class SileroVADValidator:
    """
    Sustained speech validator using Silero VAD.
    
    Implements the two-stage gate:
    1. VAD must classify frames as speech
    2. Speech must be sustained for minimum duration
    
    This prevents false positives from brief noises while maintaining responsiveness.
    """
    
    def __init__(self, vad: SileroVAD, min_speech_duration_ms: int = 400):
        """
        Initialize validator.
        
        Args:
            vad: SileroVAD instance
            min_speech_duration_ms: Minimum continuous speech duration to accept
                                   (default 400ms = 20 frames @ 20ms/frame)
        """
        self.vad = vad
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_speech_frames = int(min_speech_duration_ms / 20)  # 20ms frames
        
        # State tracking
        self.consecutive_speech_frames = 0
        self.speech_start_time: Optional[float] = None
        
        logger.info(
            f"SileroVADValidator: min_duration={min_speech_duration_ms}ms "
            f"({self.min_speech_frames} frames)"
        )
    
    def process_frame(self, audio_chunk: bytes, input_sample_rate: int = 48000) -> dict:
        """
        Process audio frame and determine if sustained speech is present.
        
        Returns:
            dict with keys:
                - is_speech: bool - Current frame classified as speech by VAD
                - is_sustained: bool - Speech sustained for minimum duration
                - duration_ms: float - Current continuous speech duration
                - frames: int - Consecutive speech frames
        """
        import time
        
        # Run VAD on frame
        is_speech = self.vad.is_speech(audio_chunk, input_sample_rate)
        
        if is_speech:
            # Speech detected - increment counter
            if self.consecutive_speech_frames == 0:
                self.speech_start_time = time.time()
            
            self.consecutive_speech_frames += 1
            
            # Calculate duration
            if self.speech_start_time:
                duration_ms = (time.time() - self.speech_start_time) * 1000
            else:
                duration_ms = 0.0
            
            # Check if sustained
            is_sustained = self.consecutive_speech_frames >= self.min_speech_frames
            
            return {
                'is_speech': True,
                'is_sustained': is_sustained,
                'duration_ms': duration_ms,
                'frames': self.consecutive_speech_frames
            }
        else:
            # No speech - reset counter
            if self.consecutive_speech_frames > 0:
                logger.debug(
                    f"Speech ended after {self.consecutive_speech_frames} frames "
                    f"({self.consecutive_speech_frames * 20}ms)"
                )
            
            self.consecutive_speech_frames = 0
            self.speech_start_time = None
            
            return {
                'is_speech': False,
                'is_sustained': False,
                'duration_ms': 0.0,
                'frames': 0
            }
    
    def reset(self):
        """Reset validator state."""
        self.consecutive_speech_frames = 0
        self.speech_start_time = None
        logger.debug("SileroVADValidator reset")
