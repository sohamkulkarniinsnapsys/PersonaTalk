"""
Audio Conditioning Pipeline for STT Accuracy

This module implements professional-grade audio pre-processing before STT.
The Sarvam STT model itself is accurate; poor transcriptions are caused by
noisy, clipped, or incomplete audio buffers.

Pipeline Steps (MANDATORY before STT):
1. DC Offset Removal
2. High-Pass Filter (80-100 Hz) - remove hum/rumble
3. Low-Pass Filter (7-8 kHz) - remove hiss/artifacts
4. Noise Floor Estimation (continuous learning)
5. Spectral Noise Suppression
6. Adaptive Gain Control (AGC)
7. Speech-Band Emphasis (300-3400 Hz)
8. Silence Trimming (edges only, preserve internal pauses)
"""

import numpy as np
import logging
from typing import Tuple, Optional
from scipy import signal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioMetrics:
    """Structured metrics for debugging and monitoring."""
    raw_rms: float
    conditioned_rms: float
    noise_floor: float
    peak_amplitude: float
    duration_ms: float
    clipping_detected: bool
    snr_db: float  # Signal-to-Noise Ratio
    spectral_flatness: float  # 0=pure tone, 1=white noise


class NoiseFloorEstimator:
    """Continuously learns background noise profile from non-speech frames."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.noise_rms_history = []
        self.max_history_len = 50  # Keep last 50 frames (~1 second)
        self.learning_rate = 0.1
        
    def update(self, frame: np.ndarray, is_speech: bool):
        """
        Update noise profile from non-speech frames.
        
        Args:
            frame: Audio samples (float32, [-1, 1])
            is_speech: Whether this frame contains speech
        """
        rms = np.sqrt(np.mean(frame ** 2))
        
        if not is_speech and rms > 0:
            # Only learn from non-speech frames
            self.noise_rms_history.append(rms)
            if len(self.noise_rms_history) > self.max_history_len:
                self.noise_rms_history.pop(0)
            
            # Update noise floor as moving average
            if self.noise_profile is None:
                self.noise_profile = rms
            else:
                self.noise_profile = (1 - self.learning_rate) * self.noise_profile + self.learning_rate * rms
    
    def get_noise_floor(self) -> float:
        """Get current noise floor estimate."""
        if self.noise_profile is None:
            # Default to very low noise if no profile yet
            return 0.001
        return max(self.noise_profile, 0.001)  # Never return zero


class AudioConditioner:
    """
    Professional-grade audio conditioning pipeline for STT accuracy.
    
    This class applies 8 critical pre-processing steps to raw audio before
    sending to STT, dramatically improving transcription accuracy.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.noise_estimator = NoiseFloorEstimator(sample_rate)
        
        # Design filters once at initialization for efficiency
        self._design_filters()
        
        # AGC state
        self.target_rms = 0.15  # Target RMS level (moderate volume)
        self.agc_attack = 0.1   # How quickly to increase gain
        self.agc_release = 0.3  # How quickly to decrease gain
        self.current_gain = 1.0
        
    def _design_filters(self):
        """Design bandpass filters for audio conditioning."""
        nyquist = self.sample_rate / 2
        
        # High-pass filter: remove low-frequency noise (hum, rumble, mic handling)
        # 80 Hz cutoff, 4th order Butterworth
        self.hp_cutoff = 80
        self.hp_b, self.hp_a = signal.butter(4, self.hp_cutoff / nyquist, btype='high')
        
        # Low-pass filter: remove high-frequency hiss and artifacts
        # 7500 Hz cutoff (preserve consonants but remove hiss)
        self.lp_cutoff = 7500
        self.lp_b, self.lp_a = signal.butter(4, self.lp_cutoff / nyquist, btype='low')
        
        # Speech-band emphasis filter: boost 300-3400 Hz (telephone quality range)
        # This is where most linguistic information lives
        self.speech_low = 300
        self.speech_high = 3400
        self.speech_b, self.speech_a = signal.butter(
            2,
            [self.speech_low / nyquist, self.speech_high / nyquist],
            btype='band'
        )
    
    def condition(self, audio_bytes: bytes, is_speech: bool = True) -> Tuple[bytes, AudioMetrics]:
        """
        Apply full conditioning pipeline to raw audio.
        
        Args:
            audio_bytes: Raw PCM s16 mono audio
            is_speech: Whether this buffer likely contains speech (for noise learning)
            
        Returns:
            (conditioned_bytes, metrics): Conditioned audio + debugging metrics
        """
        # Convert bytes to float32 [-1, 1]
        samples_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(samples_int16) == 0:
            return audio_bytes, self._empty_metrics()
        
        samples = samples_int16.astype(np.float32) / 32767.0
        
        # Capture raw metrics
        raw_rms = np.sqrt(np.mean(samples ** 2))
        raw_peak = np.max(np.abs(samples))
        
        # 1. DC Offset Removal
        samples = self._remove_dc_offset(samples)
        
        # 2. High-Pass Filter (remove hum/rumble)
        samples = signal.lfilter(self.hp_b, self.hp_a, samples)
        
        # 3. Low-Pass Filter (remove hiss)
        samples = signal.lfilter(self.lp_b, self.lp_a, samples)
        
        # 4. Update Noise Floor (continuous learning from non-speech)
        self.noise_estimator.update(samples, is_speech)
        noise_floor = self.noise_estimator.get_noise_floor()
        
        # 5. Spectral Noise Suppression (soft gating)
        samples = self._suppress_noise(samples, noise_floor)
        
        # 6. Adaptive Gain Control (AGC)
        samples, applied_gain = self._apply_agc(samples)
        
        # 7. Speech-Band Emphasis (boost linguistic frequencies)
        # Apply gentle boost to speech band
        speech_emphasized = signal.lfilter(self.speech_b, self.speech_a, samples)
        # Mix 70% emphasized + 30% original to avoid over-processing
        samples = 0.7 * speech_emphasized + 0.3 * samples
        
        # 8. Silence Trimming (edges only, preserve internal pauses)
        samples = self._trim_edges(samples, threshold=noise_floor * 3)
        
        # Final safety: prevent clipping
        peak = np.max(np.abs(samples))
        clipping_detected = False
        if peak > 0.95:
            samples = samples * (0.95 / peak)
            clipping_detected = True
            logger.debug(f"🔇 Clipping prevention: scaled by {0.95/peak:.3f}")
        
        # Convert back to int16
        samples_int16 = (samples * 32767).astype(np.int16)
        conditioned_bytes = samples_int16.tobytes()
        
        # Compute final metrics
        conditioned_rms = np.sqrt(np.mean(samples ** 2))
        snr_db = 20 * np.log10(conditioned_rms / max(noise_floor, 1e-6)) if conditioned_rms > 0 else -60
        spectral_flatness = self._compute_spectral_flatness(samples)
        
        metrics = AudioMetrics(
            raw_rms=float(raw_rms),
            conditioned_rms=float(conditioned_rms),
            noise_floor=float(noise_floor),
            peak_amplitude=float(np.max(np.abs(samples))),
            duration_ms=len(samples) / self.sample_rate * 1000,
            clipping_detected=clipping_detected,
            snr_db=float(snr_db),
            spectral_flatness=float(spectral_flatness)
        )
        
        return conditioned_bytes, metrics
    
    def _remove_dc_offset(self, samples: np.ndarray) -> np.ndarray:
        """Remove DC bias (microphone offset)."""
        return samples - np.mean(samples)
    
    def _suppress_noise(self, samples: np.ndarray, noise_floor: float) -> np.ndarray:
        """
        Soft noise gating: attenuate samples below noise threshold.
        
        This is NOT spectral subtraction (which requires FFT).
        Instead, we use soft gating in time domain - simple but effective.
        """
        # Compute RMS in small windows
        window_size = int(0.02 * self.sample_rate)  # 20ms windows
        result = samples.copy()
        
        for i in range(0, len(samples), window_size):
            window = samples[i:i+window_size]
            if len(window) == 0:
                continue
            
            window_rms = np.sqrt(np.mean(window ** 2))
            
            # Soft gate: if RMS is below 2x noise floor, attenuate
            if window_rms < noise_floor * 2:
                # Soft attenuation (not complete mute)
                attenuation = 0.3  # Keep 30% to preserve very quiet speech
                result[i:i+window_size] = window * attenuation
        
        return result
    
    def _apply_agc(self, samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Adaptive Gain Control: normalize loudness dynamically.
        
        Unlike fixed +3dB gain, AGC adapts to actual signal level.
        Prevents clipping while ensuring audibility.
        """
        current_rms = np.sqrt(np.mean(samples ** 2))
        
        if current_rms < 1e-6:
            # Silence, don't change gain
            return samples, self.current_gain
        
        # Desired gain to reach target RMS
        desired_gain = self.target_rms / current_rms
        
        # Smooth gain changes (attack/release)
        if desired_gain > self.current_gain:
            # Increasing gain (attack)
            self.current_gain = (1 - self.agc_attack) * self.current_gain + self.agc_attack * desired_gain
        else:
            # Decreasing gain (release)
            self.current_gain = (1 - self.agc_release) * self.current_gain + self.agc_release * desired_gain
        
        # Limit gain range to prevent over-amplification
        self.current_gain = np.clip(self.current_gain, 0.5, 4.0)
        
        return samples * self.current_gain, float(self.current_gain)
    
    def _trim_edges(self, samples: np.ndarray, threshold: float) -> np.ndarray:
        """
        Trim leading and trailing silence, preserve internal pauses.
        
        Args:
            samples: Audio samples
            threshold: Energy threshold below which is considered silence
        """
        # Find first non-silent sample
        abs_samples = np.abs(samples)
        window_size = int(0.02 * self.sample_rate)  # 20ms windows
        
        start_idx = 0
        for i in range(0, len(samples), window_size):
            window = abs_samples[i:i+window_size]
            if len(window) > 0 and np.mean(window) > threshold:
                start_idx = max(0, i - window_size)  # Keep one window before speech
                break
        
        # Find last non-silent sample
        end_idx = len(samples)
        for i in range(len(samples) - window_size, 0, -window_size):
            window = abs_samples[i:i+window_size]
            if len(window) > 0 and np.mean(window) > threshold:
                end_idx = min(len(samples), i + 2 * window_size)  # Keep one window after speech
                break
        
        if start_idx >= end_idx:
            # All silence or malformed
            return samples
        
        return samples[start_idx:end_idx]
    
    def _compute_spectral_flatness(self, samples: np.ndarray) -> float:
        """
        Compute spectral flatness (0=tonal, 1=noise-like).
        
        This helps identify if audio is speech (low flatness) or noise (high flatness).
        """
        if len(samples) < 512:
            return 0.5
        
        # Take FFT of first 512 samples
        fft = np.abs(np.fft.rfft(samples[:512]))
        
        # Avoid log(0)
        fft = fft + 1e-10
        
        # Spectral flatness = geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(fft)))
        arithmetic_mean = np.mean(fft)
        
        if arithmetic_mean < 1e-10:
            return 1.0
        
        flatness = geometric_mean / arithmetic_mean
        return min(flatness, 1.0)
    
    def _empty_metrics(self) -> AudioMetrics:
        """Return empty metrics for zero-length audio."""
        return AudioMetrics(
            raw_rms=0.0,
            conditioned_rms=0.0,
            noise_floor=0.0,
            peak_amplitude=0.0,
            duration_ms=0.0,
            clipping_detected=False,
            snr_db=-60.0,
            spectral_flatness=1.0
        )


def log_audio_metrics(metrics: AudioMetrics, prefix: str = ""):
    """Log structured audio metrics for debugging."""
    logger.info(f"🎧 {prefix}Audio Metrics:")
    logger.info(f"   Duration: {metrics.duration_ms:.0f}ms")
    logger.info(f"   Raw RMS: {metrics.raw_rms:.4f} → Conditioned RMS: {metrics.conditioned_rms:.4f}")
    logger.info(f"   Noise Floor: {metrics.noise_floor:.4f} | SNR: {metrics.snr_db:.1f} dB")
    logger.info(f"   Peak: {metrics.peak_amplitude:.3f} | Clipping: {metrics.clipping_detected}")
    logger.info(f"   Spectral Flatness: {metrics.spectral_flatness:.3f} (0=speech, 1=noise)")
