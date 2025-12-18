"""
Pre-warm TTS model on startup to avoid long delays on first greeting.
This is called once at server startup to load the Coqui TTS model
so that the first user's greeting completes quickly.
"""

import logging
import os
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


async def warmup_tts():
    """Pre-load Coqui TTS model asynchronously to avoid blocking startup."""
    try:
        logger.info("🔥 Pre-warming TTS model...")
        
        # Use thread pool to avoid blocking the event loop
        await sync_to_async(_warmup_tts_sync, thread_sensitive=False)()
        
        logger.info("✅ TTS model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"⚠️ TTS pre-warm failed (will initialize on first use): {e}")


def _warmup_tts_sync():
    """Synchronous TTS warm-up (runs in thread pool)."""
    try:
        from ai_personas.tts_providers import CoquiTTSProvider
        
        # Instantiate provider to trigger lazy loading
        provider = CoquiTTSProvider()
        
        # Trigger model load by calling _get_tts()
        _ = provider._get_tts()
        
    except Exception as e:
        logger.warning(f"TTS warm-up exception: {e}")
        raise
