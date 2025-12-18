## AI Coding Agent Instructions

**Project**: One-on-one AI video conferencing. Django 5.2 + Channels/Daphne + aiortc backend; Next.js 16 / React 19 frontend. Default AI is mock; Groq LLM + Sarvam/Whisper STT + Coqui TTS are wired for live mode.

### Quick Start
**Windows**: `./setup.ps1` | **Linux/macOS**: `chmod +x setup.sh && ./setup.sh`

Then in separate terminals:
- Backend: `cd backend && daphne -p 8000 config.asgi:application` (ASGI required for WebRTC)
- Frontend: `cd frontend && npm run dev`
- Visit: http://localhost:3000/dashboard (login required)

Key env vars: `NEXT_PUBLIC_API_URL=http://localhost:8000`, `AI_MODE=mock|live`, `STT_PROVIDER=sarvam|whisper|groq`

### Architecture Layers

**Data Flow**: User audio  VAD (20ms frames)  STT  Orchestrator  LLM  TTS (20ms chunks)  AI audio out

**Three Conversation Components**:
1. **ConversationPhase Controller** ([backend/ai_agent/conversation.py](backend/ai_agent/conversation.py)): Single source-of-truth for states (`GREETING  WAIT_FOR_USER  PROCESSING_USER  AI_SPEAKING`). Serializes all transitions with async lock. Rejects input until greeting finishes. Tracks `expected_user_utterance_id` to drop stale VAD buffers from prior turns.

2. **AIOrchestrator** ([backend/ai_agent/ai_orchestrator.py](backend/ai_agent/ai_orchestrator.py)): Wraps STT/LLM/TTS providers, maintains per-room history (capped 20 turns), builds prompts, truncates LLM replies at sentence boundaries. Used by `handle_utterance()`.

3. **WebRTCManager** ([backend/ai_agent/webrtc.py](backend/ai_agent/webrtc.py)): One per room, manages RTCPeerConnection, applies incoming VAD, streams TTS audio. Queues TTS as 20ms/1920-byte chunks via `queue_audio_output` (sets `ai_is_speaking` to mute VAD during playback).

**Signaling**: WebSocket at `/ws/signaling/{room_id}/` ([backend/ai_agent/routing.py](backend/ai_agent/routing.py), [backend/ai_agent/consumers.py](backend/ai_agent/consumers.py)). Serializes offer/answer; manager stays alive until last WS disconnects.

**Frontend WebRTC**: [frontend/app/hooks/useWebRTC.ts](frontend/app/hooks/useWebRTC.ts) fetches room/persona, builds offer, manages PC lifecycle. Keeps PC alive even if WS closes. Uses Google STUN.

### Key Patterns & Implementation Details

**Persona System**: DB-driven configs ([backend/ai_personas/builder.py](backend/ai_personas/builder.py)) with STRICT schema (display_name, system_prompt, flow: `interview|assistant`, voice settings). Resolved per-room via `database_sync_to_async` on offer. Defaults to `default` persona if missing.

**Providers** ([backend/ai_agent/providers.py](backend/ai_agent/providers.py)): Factory pattern (`get_providers()` returns dict of STT/LLM/TTS). 
- **STT**: MockSTT (returns canned lines, still analyzes energy), SimpleKeywordSTT, Sarvam, Whisper, Groq. Switch via `STT_PROVIDER` env var.
- **LLM**: MockLLM (echoes), Groq live (see [backend/ai_agent/live_providers/groq_llm.py](backend/ai_agent/live_providers/groq_llm.py))
- **TTS**: Coqui default (from [backend/ai_personas/tts_providers.py](backend/ai_personas/tts_providers.py)), requires ffmpeg + TTS package

**Question Bank**: Hardcoded in [backend/ai_agent/conversation.py](backend/ai_agent/conversation.py) (general/javascript/python, basic/moderate/advanced). Used by interview flow.

**Audio Contract**: Incoming = 48kHz s16 mono. VAD thresholds: start 500, continue 250, silence 600ms, min utterance 300ms. Sarvam internally resamples to 16k.

### Testing & Validation

**Run tests**: `cd backend && python -m pytest` (config: [backend/pytest.ini](backend/pytest.ini))
- Unit tests: [backend/ai_personas/tests/test_builder.py](backend/ai_personas/tests/test_builder.py), [backend/ai_agent/tests/test_persona_layer.py](backend/ai_agent/tests/test_persona_layer.py)
- Flow test: [backend/test_conversation_flow.py](backend/test_conversation_flow.py) (async conversation end-to-end)

**WebRTC Debug**: Verify WS connects to `ws://localhost:8000/ws/signaling/{room_id}/` | Check browser DevTools for audio tracks | Logs show VAD energy and phase transitions.

### Critical Caveats
- **No multiple offers per room**: Manager reuses existing PC. Creating a second offer breaks signaling.
- **Await `queue_audio_output`**: If not awaited, VAD doesn't mute during AI speech  echo loop.
- **Never bypass ConversationPhase controller**: Prevents double speak and stale utterance handling.
- **Backend config changes**: Update both frontend API base URL AND WebSocket URL in [frontend/app/hooks/useWebRTC.ts](frontend/app/hooks/useWebRTC.ts).
- **ASGI required**: Must use `daphne` (or gunicorn-asgi), not Django runserver. WebRTC needs proper async.
