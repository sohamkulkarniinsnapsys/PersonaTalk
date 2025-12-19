# AI Coding Agent Instructions

Project: One-on-one AI video conferencing. Backend: Django 5.2 + Channels/Daphne + aiortc; Frontend: Next.js 16 / React 19. Mock mode works out of the box; live providers (Groq LLM, Sarvam/Whisper STT, Coqui TTS) are wired.

## Quick Start
- Windows: `./setup.ps1`
- Linux/macOS: `chmod +x setup.sh && ./setup.sh`
- Run backend (ASGI required): `cd backend && daphne -p 8000 config.asgi:application`
- Run frontend: `cd frontend && npm run dev`
- Visit: http://localhost:3000/dashboard (login required)
- Env: `NEXT_PUBLIC_API_URL=http://localhost:8000`, `AI_MODE=mock|live`, `STT_PROVIDER=sarvam|whisper|groq`

## Architecture Overview
- **ConversationPhase Controller**: Single source-of-truth and turn-taking gate in [backend/ai_agent/conversation.py](../backend/ai_agent/conversation.py) with phases: `GREETING`, `WAIT_FOR_USER`, `PROCESSING_USER`, `AI_SPEAKING`. Holds `expected_user_utterance_id`, drops stale VAD.
- **AIOrchestrator**: STT/LLM/TTS glue in [backend/ai_agent/ai_orchestrator.py](../backend/ai_agent/ai_orchestrator.py). Keeps per-room history (20 turns), builds prompts, truncates LLM replies.
- **WebRTCManager**: Peer connection + audio I/O in [backend/ai_agent/webrtc.py](../backend/ai_agent/webrtc.py). Applies incoming VAD; streams TTS via `queue_audio_output` (20ms/1920-byte chunks) and sets `ai_is_speaking` to mute VAD.
- **Signaling**: Django Channels consumer at `/ws/signaling/{room_id}/` via [backend/ai_agent/routing.py](../backend/ai_agent/routing.py) and [backend/ai_agent/consumers.py](../backend/ai_agent/consumers.py).
- **Frontend WebRTC**: Offer/answer and PC lifecycle in [frontend/app/hooks/useWebRTC.ts](../frontend/app/hooks/useWebRTC.ts). Uses Google STUN; keeps PC alive even if WS closes.

## Personas & Providers
- **Persona System**: Strict schema + DB-backed configs resolved per room in [backend/ai_personas/builder.py](../backend/ai_personas/builder.py). Defaults to `default` persona.
- **Providers**: Factory in [backend/ai_agent/providers.py](../backend/ai_agent/providers.py) returns STT/LLM/TTS implementations.
	- STT: MockSTT, SimpleKeywordSTT, Sarvam, Whisper, Groq; select via `STT_PROVIDER`.
	- LLM: MockLLM, Groq live at [backend/ai_agent/live_providers/groq_llm.py](../backend/ai_agent/live_providers/groq_llm.py).
	- TTS: Coqui (see [backend/ai_personas/tts_providers.py](../backend/ai_personas/tts_providers.py)); requires FFmpeg + TTS packages.

## Audio & VAD Contract
- Input: 48kHz s16 mono.
- **VAD Architecture**: Turn-based interaction (2.5s silence threshold, no partial STT)
- **Single STT call per turn**: Only ONE transcription after complete 2.5s silence
- **No mid-speech STT**: All partial/streaming calls removed for accuracy
- STT: Sarvam internally resamples to 16k.
- TTS streaming: enqueue 20ms frames; always `await queue_audio_output`.
- Audio conditioning: 8-step pipeline AFTER finalization (DC removal, filters, AGC, etc.)

## Critical Patterns & Gotchas
- Use **ASGI server**: `daphne` (or gunicorn-asgi). Do not use Django `runserver`.
- **No multiple offers per room**: Manager reuses existing `RTCPeerConnection`; second offer breaks signaling.
- **Await `queue_audio_output`**: Ensures VAD mutes during AI speech; prevents echo loop.
- **Do not bypass `ConversationPhase`**: All transitions must go through the controller.
- **Update both URLs** on config changes: API base and WS in [frontend/app/hooks/useWebRTC.ts](../frontend/app/hooks/useWebRTC.ts).

## Developer Workflows
- Tests: `cd backend && python -m pytest` (see [backend/pytest.ini](../backend/pytest.ini)).
	- Unit: [backend/ai_personas/tests/test_builder.py](../backend/ai_personas/tests/test_builder.py), [backend/ai_agent/tests/test_persona_layer.py](../backend/ai_agent/tests/test_persona_layer.py)
	- Flow (async): [backend/test_conversation_flow.py](../backend/test_conversation_flow.py)
- WebRTC debug: verify WS `ws://localhost:8000/ws/signaling/{room_id}/`, check audio tracks in browser DevTools, log VAD energy and phase transitions.
- Frontend dev: Next.js app under [frontend/app](../frontend/app), components in [frontend/components](../frontend/components).

## Where to Extend
- Add STT/LLM/TTS: implement provider and register in [backend/ai_agent/providers.py](../backend/ai_agent/providers.py).
- Persona voices/prompts: update via DB or admin UI; code references in [backend/ai_personas](../backend/ai_personas).
- Room signaling/business logic: [backend/ai_agent/consumers.py](../backend/ai_agent/consumers.py), [backend/ai_agent/webrtc.py](../backend/ai_agent/webrtc.py).

Keep changes minimal and respect existing async boundaries and phase control. When unsure, follow usage patterns found in the referenced files and prefer expanding provider factories over inserting one-off integrations.
