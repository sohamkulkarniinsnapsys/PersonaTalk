import asyncio
import logging
import time
from typing import AsyncIterator
from av import AudioFrame
from aiortc import MediaStreamTrack
import numpy as np

logger = logging.getLogger(__name__)

class TTSStreamTrack(MediaStreamTrack):
    """
    An AudioStreamTrack that consumes an async iterator of PCM audio chunks
    and yields AudioFrames for WebRTC playback.
    """
    kind = "audio"

    def __init__(self, track_iterator: AsyncIterator[bytes]):
        super().__init__()
        self.track_iterator = track_iterator
        self.sample_rate = 48000
        self.channels = 1
        self.pts = 0
        # 20ms at 48kHz = 960 samples
        self.samples_per_frame = 960
        self.start_time = None
        self._ended = False

    async def recv(self):
        """
        Called by aiortc to get the next frame.
        """
        if self._ended:
            # If stream ended, keep sending silence or raise MediaStreamError?
            # Usually we should stop the track or let the peer know.
            # But valid media tracks might just stay silent.
            # Let's clean up or stop.
            self.stop()
            raise Exception("Stream ended")

        if self.start_time is None:
            self.start_time = time.time()

        try:
            # Use asyncio.wait_for to prevent hanging if iterator stalls indefinitely
            # But iterator should yield or end.
            chunk = await self.track_iterator.__anext__()
            
            # Create AudioFrame
            # Assumes chunk is 16-bit PCM mono
            # Convert bytes to numpy array (int16)
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            
            # Reshape for av: (channels, samples) -> (1, 960)
            audio_data = audio_data.reshape(1, -1)
            
            frame = AudioFrame.from_ndarray(audio_data, format='s16', layout='mono')
            frame.sample_rate = self.sample_rate
            frame.pts = self.pts
            frame.time_base = 1 / self.sample_rate
            
            self.pts += frame.samples
            return frame

        except StopAsyncIteration:
            logger.info("TTS stream iteration finished.")
            self._ended = True
            self.stop()
            raise Exception("Stream finished")
        except Exception as e:
            logger.error(f"TTSStreamTrack error: {e}")
            self._ended = True
            raise
