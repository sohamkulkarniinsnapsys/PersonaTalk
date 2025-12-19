import wave, struct, math, sys
from pathlib import Path

paths = [
    Path(r"C:\insnapsys\video_conf\backend\test_recordings\f09c61c4-6b6a-4b71-a61b-1c8a0c39dbdd\1766120797269__utt-1__23080ms.wav"),
    Path(r"C:\insnapsys\video_conf\backend\test_recordings\f09c61c4-6b6a-4b71-a61b-1c8a0c39dbdd\1766120827115__utt-2__30680ms.wav"),
]

for p in paths:
    with wave.open(str(p), 'rb') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        rms_acc = 0.0
        peak = 0
        count = 0
        dc_acc = 0.0
        chunk = 48000
        while True:
            data = w.readframes(chunk)
            if not data:
                break
            s_count = len(data) // sampwidth
            fmt = '<' + ('h' * s_count if sampwidth == 2 else 'b' * s_count)
            samples = struct.unpack(fmt, data)
            if n_channels > 1:
                samples = samples[::n_channels]
            for s in samples:
                peak = max(peak, abs(s))
                rms_acc += s*s
                dc_acc += s
                count += 1
        rms = math.sqrt(rms_acc / max(count, 1)) if count else 0.0
        dc = (dc_acc / max(count, 1)) if count else 0.0
        norm_rms = rms / 32767.0
        norm_peak = peak / 32767.0
        norm_dc = dc / 32767.0
        duration_s = nframes / float(framerate)
        print({
            'path': str(p),
            'channels': n_channels,
            'samplerate': framerate,
            'frames': nframes,
            'duration_s': round(duration_s, 2),
            'rms_norm': round(norm_rms, 6),
            'peak_norm': round(norm_peak, 6),
            'dc_offset_norm': round(norm_dc, 6),
        })
