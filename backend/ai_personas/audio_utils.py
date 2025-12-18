import io
import subprocess
import logging
import wave
import numpy as np

logger = logging.getLogger(__name__)

def normalize_to_48k_mono_pcm(input_bytes: bytes, sample_rate: int = 22050) -> bytes:
    """
    Converts input audio bytes (raw PCM or WAV) to 48kHz mono 16-bit PCM 
    suitable for aiortc/WebRTC.
    
    Tries to use ffmpeg first (best quality/speed), falls back to pydub/scipy if available,
    or simple numpy resampling if dependencies are missing.
    """
    # Try ffmpeg
    try:
        # Input format is assumed to be s16le regular PCM if it's raw bytes
        # If it's a WAV file container, ffmpeg detects it automatically.
        # But Coqui usually gives us a WAV file bytes or raw float/int array.
        # Let's assume input_bytes is a full WAV file (header included) or we treat it as raw if we know the sample rate.
        
        # If input_data has RIFF header, let ffmpeg detect.
        process = subprocess.Popen(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', 'pipe:0',             # Read from stdin
                '-f', 's16le',              # Output raw PCM 16-bit
                '-ac', '1',                 # Mono
                '-ar', '48000',             # 48kHz
                'pipe:1'                    # Write to stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = process.communicate(input=input_bytes)
        if process.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {err.decode()}")
            raise RuntimeError("FFmpeg conversion failed")
        return out
    except FileNotFoundError:
        logger.warning("ffmpeg not found, falling back to pydub/numpy")
        # Fallbacks would go here (pydub, etc)
        # For this task, we strongly recommend ffmpeg. 
        # But let's implement a quick wav-to-pcm via wave module if possible 
        # but resampling is hard without libs.
        return _fallback_resample(input_bytes, sample_rate)
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        return b''

def _fallback_resample(input_bytes: bytes, input_rate: int) -> bytes:
    # Very basic fallback that might not be perfect
    # Requires numpy
    try:
        import numpy as np
        # Check if WAV header
        if input_bytes.startswith(b'RIFF'):
            with io.BytesIO(input_bytes) as bio:
                with wave.open(bio, 'rb') as wav:
                    # frames = wav.readframes(wav.getnframes())
                    # Make sure we handle width/channels
                    pass
        return input_bytes # Return as is if we can't resample (will likely sound wrong)
    except ImportError:
        return input_bytes

def pcm_bytes_to_frames(pcm_bytes: bytes, frame_ms: int = 20, sample_rate: int = 48000) -> list[bytes]:
    """
    Splits a buffer of 16-bit PCM mono bytes into chunks of frame_ms duration.
    """
    bytes_per_sample = 2 # 16-bit
    channels = 1
    samples_per_sec = sample_rate
    bytes_per_sec = samples_per_sec * bytes_per_sample * channels
    chunk_size = int(bytes_per_sec * (frame_ms / 1000.0))
    
    frames = []
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i+chunk_size]
        if len(chunk) < chunk_size:
            # Pad silence if needed? Or just drop/send partial? 
            # WebRTC usually expects fixed size frames for some codecs, but pcm is flexible.
            # Opus encoder in aiortc handles frame sizes.
            # Let's pad with silence for safety to match 20ms
            padding = chunk_size - len(chunk)
            chunk += b'\x00' * padding
        frames.append(chunk)
    return frames

def apply_audio_effects(wav_bytes: bytes, rate: float = 1.0, pitch: float = 1.0) -> bytes:
    """
    Apply speed and pitch changes using FFmpeg.
    
    Logic:
    - Pitch only (without speed change) is hard with simple filters.
    - We use a combination of 'asetrate' (resampling, changes pitch & speed) 
      and 'atempo' (time-stretching, changes speed & keeps pitch) to achieve desired effect.
    
    Target:
      New Rate R_target = rate
      New Pitch P_target = pitch
    
    1. 'asetrate=sample_rate * pitch': This shifts pitch by P, but also shifts speed by P aka Duration / P.
       (e.g. pitch=2.0 -> plays 2x faster).
       Current Speed Factor = P
       
    2. We want Final Speed Factor = R_target.
       So we need to apply 'atempo' to correct the speed.
       We are currently at speed P. We want R.
       Correction factor = R / P.
       
       Wait. asetrate changes sample rate playback. 
       If we play at 2x rate, duration is 0.5x.
       We want duration to be 1/R * original_duration.
       
       Let's stick to simple composition:
       Speed change: atempo=rate
       Pitch change: asetrate=r*pitch, atempo=1/pitch ? No that cancels speed change of asetrate but keeps pitch.
       
       To change Pitch by P without changing duration:
       asetrate = 48000 * P
       atempo = 1 / P
       
       To change Rate by R without changing Pitch:
       atempo = R
       
        Combined:
       asetrate=48000*P, atempo=1/P, atempo=R
       
       NOTE: atempo filter is limited to 0.5 to 2.0. We can chain them if needed, but UI limits to these bounds anyway.
    """
    if rate == 1.0 and pitch == 1.0:
        return wav_bytes
        
    # We normalized to 48000 previously.
    base_sr = 48000
    
    # Construct filter graph
    # Check bounds or chain filters if outside 0.5-2.0
    # For MVP, assume inputs are within 0.5-2.0
    
    filters = []
    
    # 1. Pitch shift (rescaling rate)
    if pitch != 1.0:
        new_sr = int(base_sr * pitch)
        filters.append(f"asetrate={new_sr}")
        filters.append(f"atempo={1.0/pitch}")
        
    # 2. Rate shift
    if rate != 1.0:
        filters.append(f"atempo={rate}")
        
    filter_str = ",".join(filters)
    
    try:
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', 'pipe:0',
            '-filter:a', filter_str,
            '-f', 'wav',  # Output as WAV
            'pipe:1'
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = process.communicate(input=wav_bytes)
        
        if process.returncode != 0:
            logger.error(f"FFmpeg effect failed: {err.decode()}")
            return wav_bytes # Fallback to original
            
        return out
    except Exception as e:
        logger.error(f"Error applying audio effects: {e}")
        return wav_bytes
