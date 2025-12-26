# AI Coding Agent Instructions (Concise)

Purpose: Real-time, turn-based AI voice calls with intelligent barge-in.

Stack: Django + Channels/Daphne + aiortc (backend), Next.js (frontend).

## Run & Test
- Backend (ASGI only): `cd backend && daphne -p 8000 config.asgi:application`
- Frontend: `cd frontend && npm run dev`
- Windows setup: `./setup.ps1` (installs deps; ensure FFmpeg)
- Key tests: `cd backend && python -m pytest test_deep_fixes.py -v` and `python test_audio_conditioning.py`

## Environment
- Backend env file: [backend/.env](backend/.env)
  - `AI_MODE=mock|live`, `STT_PROVIDER=sarvam|whisper|groq`, `GROQ_API_KEY`, `SARVAM_API_KEY`, `TTS_PROVIDER=coqui`
- Frontend env file: [frontend/.env.local](frontend/.env.local)
  - `NEXT_PUBLIC_API_URL=http://localhost:8000`

## Architecture & Key Files
- WebRTC + VAD: [backend/ai_agent/webrtc.py](backend/ai_agent/webrtc.py)
  - 48kHz s16 mono, 20ms frames; 2.5s silence → turn end; strict ASGI.
- Conversation state: [backend/ai_agent/conversation.py](backend/ai_agent/conversation.py)
  - `WAIT_FOR_USER` → `VALIDATING_UTTERANCE` → `PROCESSING_USER` → `AI_SPEAKING`; single STT per turn.
- AI orchestration: [backend/ai_agent/ai_orchestrator.py](backend/ai_agent/ai_orchestrator.py)
  - Provider abstraction for STT/LLM/TTS; persona-scoped history.
- Barge-in (persona-aware): [backend/ai_agent/barge_in_state_machine.py](backend/ai_agent/barge_in_state_machine.py), [backend/ai_agent/persona_behaviors.py](backend/ai_agent/persona_behaviors.py)
  - Stage 1 (speech energy/SNR) → Stage 2 (quick STT + heuristics); stop-words allowed mid TTS.
- Audio conditioning: [backend/ai_agent/audio_conditioning.py](backend/ai_agent/audio_conditioning.py)
  - 8-step pipeline (DC removal, HP/LP, noise suppression, AGC, speech-band emphasis) before STT.
- Personas: [backend/ai_personas/builder.py](backend/ai_personas/builder.py)
  - JSON schema, presets, voice config; `metadata.source_template_id` required.

## Service Boundaries
- WebSocket signaling: `/ws/signaling/{room_id}/` via Channels (backend).
- Frontend: Next.js App Router; auth at `/`, dashboard at `/dashboard`.
- Media: aiortc handles audio; TTS chunks streamed back (20ms).

## Project-Specific Patterns
- Use `database_sync_to_async` for ORM inside async (Channels/aiortc).
- Bind `utterance_id` at capture time; reuse after STT to avoid race conditions.
- Enforce 48kHz s16 mono; resample if needed; frames are 960 samples.
- Turn-based only: no partial transcripts; single STT call per user turn.
- Barge-in runs during AI speech; cancels TTS on valid interruption.

## Critical Workflows & Debugging
- Logs: watch for emoji `logger.info` markers (🔊, ✅, ❌, 🛑) around VAD/turn finalization.
- Recordings: [backend/test_recordings](backend/test_recordings) store PCM/WAV per room for analysis.
- Speaking state UI: see [frontend/app/hooks/useWebRTC.ts](frontend/app/hooks/useWebRTC.ts).
- Migrations: `python manage.py makemigrations && python manage.py migrate`.
- Common issues: never use `runserver` (WSGI), ensure FFmpeg for Coqui, set STT/TTS keys for `AI_MODE=live`.

## Integration Points
- Message types: `offer`, `answer`, `ice-candidate`, `transcript`, `agent_state`, `speaking_state`.
- Provider registry: [backend/ai_agent/providers.py](backend/ai_agent/providers.py) and [backend/ai_agent/live_providers](backend/ai_agent/live_providers).

Docs to consult: [VAD_ARCHITECTURE_FIX.md](VAD_ARCHITECTURE_FIX.md), [AUDIO_CONDITIONING_README.md](AUDIO_CONDITIONING_README.md), [INTELLIGENT_BARGE_IN_README.md](INTELLIGENT_BARGE_IN_README.md).

---

Legacy detailed guide (for reference) starts below.

# AI Coding Agent Instructions

Project: Real-time AI voice conversation system. One-on-one video conferencing with intelligent barge-in and turn-based interaction.

**Stack**: Django 5.2 + Channels/Daphne + aiortc (backend) | Next.js 16 + React 19 (frontend)

## Quick Start
- **Windows**: `./setup.ps1` | **Linux/macOS**: `chmod +x setup.sh && ./setup.sh`
- **Backend** (ASGI required): `cd backend && daphne -p 8000 config.asgi:application`
- **Frontend**: `cd frontend && npm run dev`
- **Access**: http://localhost:3000 → redirects to `/dashboard` after auth
- **Testing**: `cd backend && python -m pytest test_deep_fixes.py -v` (12 tests)

## Environment Variables (Critical)
Backend `.env` location: `backend/.env` (NOT root) - loaded by Django settings
Frontend `.env.local` location: `frontend/.env.local` - loaded by Next.js

```bash
# Backend (backend/.env)
AI_MODE=mock|live                    # mock=dev, live=production providers
STT_PROVIDER=sarvam|whisper|groq    # Speech-to-text provider
GROQ_API_KEY=your_key               # Required for live LLM
SARVAM_API_KEY=your_key             # Required for Sarvam STT
TTS_PROVIDER=coqui                   # Text-to-speech (requires FFmpeg)

# Frontend (frontend/.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000  # Backend API base URL
```

## Architecture Overview

### Core Components
1. **WebRTC Audio Pipeline** (`backend/ai_agent/webrtc.py`)
   - Peer connection lifecycle (offer/answer via WebSocket)
   - Voice Activity Detection (VAD): 2.5s silence threshold for turn-based interaction
   - Audio I/O: 48kHz s16 mono, 20ms frames (960 samples), streamed via `RTCPeerConnection`
   - **CRITICAL**: Use `database_sync_to_async` for ORM calls in async context

2. **Conversation Controller** (`backend/ai_agent/conversation.py`)
   - State machine: `WAIT_FOR_USER` → `VALIDATING_UTTERANCE` → `PROCESSING_USER` → `AI_SPEAKING`
   - **Single STT call per turn** (no partial transcripts) after 2.5s continuous silence
   - Semantic validation: blocks LLM on incomplete thoughts (ends with "and", "between", etc.)
   - **Barge-in support**: User can interrupt AI mid-speech via stop words ("stop", "wait", "excuse me")

3. **AI Orchestrator** (`backend/ai_agent/ai_orchestrator.py`)
   - STT/LLM/TTS glue layer with provider abstraction
   - Persona-isolated history (20 turns max) via `PersonaContext`
   - Prompt builder includes system prompt, examples, and conversation history

4. **Intelligent Barge-In** (`backend/ai_agent/barge_in_state_machine.py`, `persona_behaviors.py`)
   - Two-stage pipeline: Speech validation (energy/SNR) → Query validation (STT + heuristics)
   - **Persona-aware**: Technical interviewer (aggressive), Expert (conservative), Assistant (balanced)
   - Stop-word detection runs DURING AI playback (ultra-low 300ms threshold, separate from barge-in)
   - See [INTELLIGENT_BARGE_IN_README.md](../INTELLIGENT_BARGE_IN_README.md)

5. **Audio Conditioning** (`backend/ai_agent/audio_conditioning.py`)
   - 8-step pipeline: DC removal → HP/LP filters → noise suppression → AGC → speech emphasis
   - Runs AFTER utterance finalization, BEFORE STT call
   - Continuous noise floor learning from non-speech frames
   - See [AUDIO_CONDITIONING_README.md](../AUDIO_CONDITIONING_README.md)

6. **Personas** (`backend/ai_personas/builder.py`)
   - Strict JSON schema validation (see `PERSONA_SCHEMA`)
   - DB-backed configs with hardcoded presets (technical-interviewer, helpful-assistant, etc.)
   - Voice settings: provider, preset_id (p225/p226), speed, pitch, style
   - Flow modes: `interview` (AI leads) vs `assistant` (user leads)

### Service Boundaries
- **Backend**: Django ASGI app with Channels for WebSocket signaling (`/ws/signaling/{room_id}/`)
- **Frontend**: Next.js App Router with React Server Components (auth at `/`, dashboard at `/dashboard`)
- **Real-time**: aiortc handles WebRTC media, Channels handles signaling/control messages
- **Database**: SQLite (dev), models in `ai_agent.models` (Call, Room) and `ai_personas.models` (Persona)

### Data Flows
```
Browser Microphone (getUserMedia)
  ↓ WebRTC (offer/answer negotiation)
RTCPeerConnection (aiortc backend)
  ↓ 48kHz s16 mono frames
VAD Detection (2.5s silence = turn end)
  ↓ Complete utterance buffer
Audio Conditioning (8-step pipeline)
  ↓ Cleaned PCM audio
STT Provider (Sarvam/Whisper/Groq)
  ↓ Transcript text
Conversation Controller (state machine + validation)
  ↓ Complete, validated utterance
LLM Provider (Groq/Mock)
  ↓ AI response text
TTS Provider (Coqui/Mock)
  ↓ Synthesized audio (48kHz s16 mono)
Queue to RTCPeerConnection (20ms chunks)
  ↓ WebRTC streaming
Browser Speakers
```




## Critical Developer Workflows

### Running Tests
```bash
cd backend
python -m pytest                          # All tests
python -m pytest test_deep_fixes.py -v    # Voice system validation (12 tests)
python test_audio_conditioning.py         # Audio pipeline validation
```

### Starting Development Servers
```bash
# Terminal 1: Backend (MUST use Daphne, NOT Django runserver)
cd backend
daphne -p 8000 config.asgi:application

# Terminal 2: Frontend
cd frontend
npm run dev
```
**Why Daphne?** Django `runserver` doesn't support ASGI/WebSocket. Using it breaks real-time audio.

### Debugging Audio Issues
1. **Check VAD logs**: Look for "🔊 SINGLE STT CALL" in backend console (should appear ONCE per user turn)
2. **Verify environment**: `AI_MODE=mock` should work out-of-box; `AI_MODE=live` requires API keys
3. **Test recordings**: Audio saved to `backend/test_recordings/{room_id}/` as PCM+WAV for offline analysis
4. **Check speaking states**: Frontend should show `speaking_state` events for user/AI (see `useWebRTC.ts`)

### Common Build/Run Issues
- **ModuleNotFoundError**: Install deps: `pip install -r requirements.txt` (backend) or `npm install` (frontend)
- **WebSocket 403/404**: Verify Daphne is running, not Django runserver
- **Empty transcripts**: Check `STT_PROVIDER` env var and API keys; review audio conditioning metrics in logs
- **TTS fails**: Ensure FFmpeg installed system-wide (`winget install ffmpeg` on Windows)

### Database Migrations
```bash
cd backend
python manage.py makemigrations  # Generate migration files
python manage.py migrate          # Apply to database
```
**CRITICAL**: Always use `@database_sync_to_async` when accessing Django ORM in async functions (WebRTC/Channels context).

## Critical Patterns & Gotchas

### 1. ASGI Server Requirement
**Never use `python manage.py runserver`** - it only supports WSGI. Always use:
```bash
daphne -p 8000 config.asgi:application
```
This is the #1 cause of "WebSocket won't connect" issues.

### 2. Async Boundaries (Database Access)
Django ORM is synchronous but WebRTC/Channels code is async. **Always wrap ORM calls**:
```python
from asgiref.sync import sync_to_async

# WRONG - will hang/crash:
persona = Persona.objects.get(slug=slug)

# RIGHT - async wrapper:
@sync_to_async
def get_persona():
    return Persona.objects.get(slug=slug)

persona = await get_persona()
```
See examples in `webrtc.py` lines 215-240 (handle_offer method).

### 3. utterance_id Binding
**Bind utterance IDs to audio buffers at capture time**, not at STT completion:
```python
# Capture ID BEFORE async STT call
current_utterance_id = expected_id or str(uuid.uuid4())
audio_buffer = bytes(buffer)

# Later, use bound ID (not dynamic expected_id)
await controller.handle_user_utterance(transcript, current_utterance_id)
```
See `webrtc.py` lines 555-595 for production pattern. This prevents race conditions when STT takes 300-600ms.

### 4. Audio Frame Format
**All audio must be 48kHz s16 mono**:
- `s16` = 16-bit signed integer PCM
- 20ms frames = 960 samples/frame = 1920 bytes/frame
- Resampling required if mic provides 44.1kHz or other rates

See `audio_conditioning.py` for normalization pipeline.

### 5. Turn-Based vs Streaming
This system is **turn-based** (like ChatGPT voice mode), NOT continuous/streaming:
- User speaks → 2.5s silence → **ONE** STT call → AI responds
- No partial transcripts during speech
- Barge-in is explicit interruption, not parallel streams

## Common Pitfalls & Solutions

### Issue: "User gets cut off mid-sentence"
**Cause**: `SILENCE_DURATION_MS` too low (was 800ms, needs 2500ms)
**Fix**: Check `webrtc.py` line ~270, ensure `SILENCE_DURATION_MS = 2500`
**Why**: Natural pauses (1-2s) are part of utterances, not turn boundaries

### Issue: "Duplicate STT calls for same audio"
**Cause**: `utterance_in_flight` flag not set/checked
**Fix**: See `webrtc.py` lines 320-360 for correct pattern:
```python
if utterance_in_flight:
    continue  # Skip duplicate processing
utterance_in_flight = True
# ... STT call ...
finally:
    utterance_in_flight = False
```

### Issue: "utterance_id mismatch errors"
**Cause**: Using dynamic `expected_id` instead of bound ID
**Fix**: Capture `current_utterance_id = expected_id` BEFORE async STT, use captured value after

### Issue: "Empty transcripts from Sarvam/STT"
**Causes**:
1. Audio buffer too short (< 1.5s)
2. Low SNR (< -10 dB) or high spectral flatness (> 0.85)
3. Missing audio conditioning

**Fix**: Check logs for "Audio Metrics" - SNR, duration, spectral_flatness. See `AUDIO_CONDITIONING_README.md`.

### Issue: "TTS blocks user from speaking"
**Cause**: `ai_is_speaking = True` disables VAD
**Fix**: Barge-in detection now runs DURING AI playback (lines 375-420 in `webrtc.py`). User speech triggers immediate TTS cancel.

## Where to Extend

### Adding New STT/LLM/TTS Providers
1. Implement interface in `backend/ai_agent/interfaces.py`
2. Add provider class in `backend/ai_agent/live_providers/`
3. Register in `backend/ai_agent/providers.py` `get_providers()` function
4. Add env var for API key in `backend/.env`

Example: See `GroqWhisperSTT` in `live_providers/stt.py` lines 80-150.

### Adding Custom Personas
1. Define preset in `backend/ai_personas/builder.py` `PRESET_TEMPLATES` dict
2. Create DB entry via `/api/personas/` endpoint
3. **CRITICAL**: Set `metadata.source_template_id` to match preset key
4. Optionally add barge-in behavior in `persona_behaviors.py`

### Modifying WebSocket Signaling
- **Backend**: Edit `backend/ai_agent/consumers.py` (CallConsumer class)
- **Frontend**: Edit `frontend/app/hooks/useWebRTC.ts` (websocket.onmessage handler)
- **Message types**: `offer`, `answer`, `ice-candidate`, `transcript`, `agent_state`, `speaking_state`

## Documentation Map

| Topic | File | Lines | Purpose |
|-------|------|-------|---------|
| **Architecture Overview** | DEEP_FIXES_ARCHITECTURAL_CHANGES.md | 870 | Complete system design decisions |
| **VAD & Turn-Taking** | VAD_ARCHITECTURE_FIX.md | 280 | Why 2.5s silence threshold |
| **Audio Pipeline** | AUDIO_CONDITIONING_README.md | 350 | 8-step conditioning details |
| **Barge-In System** | INTELLIGENT_BARGE_IN_README.md | 450 | Two-stage interruption validation |
| **Stop-Word Detection** | STOP_WORD_IMPLEMENTATION_COMPLETE.md | 520 | Ultra-low latency control commands |
| **Persona System** | backend/ai_personas/builder.py | 450 | Schema, templates, voice configs |
| **Testing Guide** | TESTING_GUIDE.md | 280 | Production fix validation |

**Pro tip**: Search codebase for `logger.info` with emojis (🔊, ✅, ❌, 🛑) - these mark critical decision points.


