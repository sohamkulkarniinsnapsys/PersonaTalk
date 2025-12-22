# AI Coding Agent Instructions

Project: One-on-one AI video conferencing. Backend: Django 5.2 + Channels/Daphne + aiortc; Frontend: Next.js 16 / React 19. Mock mode works out of the box; live providers (Groq LLM, Sarvam/Whisper STT, Coqui TTS) are wired.

## Quick Start
- Windows: `./setup.ps1`
- Linux/macOS: `chmod +x setup.sh && ./setup.sh`
- Run backend (ASGI required): `cd backend && daphne -p 8000 config.asgi:application`
- Run frontend: `cd frontend && npm run dev`
- Visit: http://localhost:3000 (auth required, redirects to `/dashboard`)
- Env: `NEXT_PUBLIC_API_URL=http://localhost:8000`, `AI_MODE=mock|live`, `STT_PROVIDER=sarvam|whisper|groq`

### Prerequisites for Live Mode
- **FFmpeg**: Required for audio processing. Install via `winget install ffmpeg` (Windows), `brew install ffmpeg` (Mac), or `apt install ffmpeg` (Linux)
- **Coqui TTS**: `pip install TTS numpy soundfile pydub ffmpeg-python` for speech synthesis
- **API Keys**: Set `GROQ_API_KEY`, `SARVAM_API_KEY` for live providers (see `.env.example`)

### Environment Variable Priority
- Backend: `.env` file in `backend/` directory (NOT root) - loaded by Django settings
- Frontend: `.env.local` in `frontend/` directory - loaded by Next.js
- Never commit `.env` or `.env.local` files (both in `.gitignore`)

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
- **Intelligent Barge-In**: Two-stage interruption pipeline (speech validation + query validation) with persona-aware behavior switching in [backend/ai_agent/barge_in_state_machine.py](../backend/ai_agent/barge_in_state_machine.py) and [backend/ai_agent/persona_behaviors.py](../backend/ai_agent/persona_behaviors.py). See [INTELLIGENT_BARGE_IN_README.md](../INTELLIGENT_BARGE_IN_README.md) for details.

## Audio & VAD Contract
- Input: 48kHz s16 mono.
- **VAD Architecture**: Turn-based interaction (2.5s silence threshold, no partial STT)
- **Single STT call per turn**: Only ONE transcription after complete 2.5s silence
- **No mid-speech STT**: All partial/streaming calls removed for accuracy
- STT: Sarvam internally resamples to 16k.
- TTS streaming: enqueue 20ms frames; always `await queue_audio_output`.
- Audio conditioning: 8-step pipeline AFTER finalization (DC removal, filters, AGC, etc.)
- **Recorded utterances**: Saved to `backend/test_recordings/{room_id}/` as both raw PCM and WAV for debugging

## Critical Patterns & Gotchas
- Use **ASGI server**: `daphne` (or gunicorn-asgi). Do not use Django `runserver`.
- **No multiple offers per room**: Manager reuses existing `RTCPeerConnection`; second offer breaks signaling.
- **Await `queue_audio_output`**: Ensures VAD mutes during AI speech; prevents echo loop.
- **Do not bypass `ConversationPhase`**: All transitions must go through the controller.
- **Update both URLs** on config changes: API base and WS in [frontend/app/hooks/useWebRTC.ts](../frontend/app/hooks/useWebRTC.ts).
- **Async/await boundaries**: All AI processing (STT, LLM, TTS) must be awaited; never mix sync/async contexts.
- **utterance_id binding**: IDs are bound to audio buffers at capture time, NOT at STT result time (prevents race conditions).
- **State machine discipline**: ConversationPhase transitions are logged at INFO level; grep for phase changes to debug stuck states.

## Developer Workflows
- **Tests**: `cd backend && python -m pytest` (see [backend/pytest.ini](../backend/pytest.ini)).
	- Unit: [backend/ai_personas/tests/test_builder.py](../backend/ai_personas/tests/test_builder.py), [backend/ai_agent/tests/test_persona_layer.py](../backend/ai_agent/tests/test_persona_layer.py)
	- Flow (async): [backend/test_conversation_flow.py](../backend/test_conversation_flow.py)
	- Deep fixes validation: `python -m pytest test_deep_fixes.py -v` (12 tests covering VAD, utterance_id, barge-in)
- **Audio debugging**: Recorded utterances saved to `backend/test_recordings/{room_id}/` as PCM+WAV for offline analysis
- **WebRTC debug**: Verify WS `ws://localhost:8000/ws/signaling/{room_id}/`, check audio tracks in browser DevTools, log VAD energy and phase transitions
- **Frontend dev**: Next.js app under [frontend/app](../frontend/app), components in [frontend/components](../frontend/components)
- **Auth system**: Custom-branded login at `/` (root), redirects to `/dashboard` after auth (see [backend/AUTH_README.md](../backend/AUTH_README.md))
- **Docker**: Optional `docker-compose.yml` available for containerized deployment
- **Log monitoring**: Key patterns to grep for debugging:
	- `🔊 SINGLE STT CALL` - Finalization events with full context
	- `⚠️ Stale or unexpected utterance_id` - Race condition warnings
	- `🎮 Controller` - Controller state transitions
	- `🛑 BARGE-IN` - User interruption events

## Where to Extend
- Add STT/LLM/TTS: implement provider and register in [backend/ai_agent/providers.py](../backend/ai_agent/providers.py).
- Persona voices/prompts: update via DB or admin UI; code references in [backend/ai_personas](../backend/ai_personas).
- Room signaling/business logic: [backend/ai_agent/consumers.py](../backend/ai_agent/consumers.py), [backend/ai_agent/webrtc.py](../backend/ai_agent/webrtc.py).

## Common Pitfalls & Solutions

### Issue: Early Utterance Cutoff
- **Symptom**: Users cut off mid-sentence after natural pauses
- **Fix**: Check `SILENCE_DURATION_MS` is 2500ms (not 800ms)
- **Debug**: Look for "Finalization trigger: XXXms continuous silence" in logs

### Issue: Duplicate STT Calls
- **Symptom**: Same audio buffer transcribed twice (API cost spike)
- **Fix**: Verify `utterance_in_flight` flag in webrtc.py prevents concurrent STT
- **Debug**: Grep for "🔊 Calling STT" - should see exactly one per utterance

### Issue: Race Conditions with utterance_id
- **Symptom**: "Stale or unexpected utterance_id" warnings, lost utterances
- **Fix**: Ensure `current_utterance_id` bound at buffer capture, not at STT result
- **Debug**: Check logs show same utterance_id from buffer capture through processing

### Issue: WebSocket Closes but Audio Continues
- **Symptom**: Peer connection stays alive after WS disconnect
- **Fix**: This is intentional! PC lifecycle independent of WS for audio continuity
- **Debug**: Check logs - PC should only close on manager.close() or last connection drop

### Issue: Empty Transcripts from STT
- **Symptom**: "⚠️ Sarvam AI returned empty transcript" in logs
- **Fix**: Check audio quality (SNR, spectral flatness), verify MIN_BUFFER_MS >= 1500
- **Debug**: Review audio conditioning metrics in logs (SNR > 0 dB, flatness < 0.85)

## Documentation Map
Critical docs for deep understanding:
- **Architecture**: [DEEP_FIXES_ARCHITECTURAL_CHANGES.md](../DEEP_FIXES_ARCHITECTURAL_CHANGES.md) - 870+ line technical deep-dive
- **Testing**: [TESTING_GUIDE.md](../TESTING_GUIDE.md) - Production test scenarios with expected behaviors
- **Deployment**: [DEPLOYMENT_CHECKLIST.md](../DEPLOYMENT_CHECKLIST.md) - Step-by-step deployment validation
- **Audio Pipeline**: [AUDIO_CONDITIONING_README.md](../AUDIO_CONDITIONING_README.md) - 8-step processing pipeline details
- **VAD Fix**: [VAD_ARCHITECTURE_FIX.md](../VAD_ARCHITECTURE_FIX.md) - Turn-based interaction implementation
- **Intelligent Barge-In**: [INTELLIGENT_BARGE_IN_README.md](../INTELLIGENT_BARGE_IN_README.md) - Two-stage interruption pipeline with persona-aware behavior
- **All Docs**: [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md) - Complete navigation guide

Keep changes minimal and respect existing async boundaries and phase control. When unsure, follow usage patterns found in the referenced files and prefer expanding provider factories over inserting one-off integrations.
