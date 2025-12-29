# AI Coding Agent Instructions

**Purpose**: Real-time, turn-based AI voice calls with persona-aware barge-in interruption.

**Stack**: Django 5.2 + Channels/Daphne + aiortc (backend), Next.js 15 (frontend).

---

## Critical Commands

**Backend** (ASGI only):  
```bash
cd backend && daphne -p 8000 config.asgi:application
```
⚠️ **NEVER use `runserver`** — breaks WebSocket/aiortc entirely (WSGI-only).

**Frontend**: `cd frontend && npm run dev`

**Bootstrap** (installs FFmpeg + deps):
- Windows: `.\setup.ps1`
- Linux/macOS: `./setup.sh`

**Tests** (validation order):
```bash
cd backend
python -m pytest test_deep_fixes.py -v              # VAD/turn-taking FSM
python test_audio_conditioning.py                   # 8-step audio pipeline
python test_conversation_flow.py                    # State transitions
```

---

## Environment & Dependencies

**Backend** ([backend/.env](backend/.env)):
```bash
AI_MODE=mock|live                           # mock for dev; live requires API keys
STT_PROVIDER=sarvam|whisper|groq            # Speech-to-text
TTS_PROVIDER=coqui|sarvam                   # Text-to-speech
GROQ_API_KEY=<key>                          # If STT_PROVIDER=groq
SARVAM_API_KEY=<key>                        # If using Sarvam
```

**Frontend** ([frontend/.env.local](frontend/.env.local)):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**System Requirements**:  
FFmpeg must be system-wide (required for Coqui TTS):
- Windows: `winget install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

**Key Python Packages** (see [requirements.txt](backend/requirements.txt)):
- `aiortc` (WebRTC media)
- `daphne` (ASGI server)
- `channels` (WebSocket)
- `TTS` (Coqui XTTS v2 for audio synthesis)
- `torch`, `torchaudio` (GPU acceleration optional)

---

## Architecture Overview

### Core Data Model
- **Room** (`ai_agent.models`): Conversation container (owner, is_active)
- **Call**: Room session with selected Persona and transcript
- **InterviewSession**: Persistent state for interview persona (score, completion)
- **InterviewQuestionState**: Per-question lifecycle (eval, hints, scoring)

### Core Components (Backend)

1. **WebRTC Media** ([webrtc.py](backend/ai_agent/webrtc.py))
   - Audio: 48kHz s16 mono, 20ms frames (960 samples = 1920 bytes)
   - VAD: 2.5s silence (`SILENCE_DURATION_MS`) triggers turn-end
   - All async DB calls wrapped with `@database_sync_to_async`

2. **Conversation FSM** ([conversation.py](backend/ai_agent/conversation.py))  
   - States: `WAIT_FOR_USER` → `VALIDATING_UTTERANCE` → `PROCESSING_USER` → `AI_SPEAKING`
   - **Non-negotiable rule**: ONE STT call per complete turn (no partial transcripts)
   - Question bank built-in for interview persona

3. **AI Orchestration** ([ai_orchestrator.py](backend/ai_agent/ai_orchestrator.py))
   - Provider-agnostic wrapper for STT/LLM/TTS
   - Registry: [providers.py](backend/ai_agent/providers.py)
   - Live providers: [live_providers/](backend/ai_agent/live_providers/)
   - Persona-scoped history management

4. **Barge-In System** (persona-aware mid-TTS interruption)
   - State machine: [barge_in_state_machine.py](backend/ai_agent/barge_in_state_machine.py)
   - Persona behaviors: [persona_behaviors.py](backend/ai_agent/persona_behaviors.py)
   - Stop-word detection: [stop_word_interruption.py](backend/ai_agent/stop_word_interruption.py)
   - Two-stage validation: Energy/SNR → STT + heuristics

5. **Audio Conditioning** ([audio_conditioning.py](backend/ai_agent/audio_conditioning.py))
   - 8-step pipeline: DC removal → HP filter → LP filter → noise suppression → AGC → speech emphasis → peak norm → clip prevention
   - Adaptive room noise learning
   - Metrics: SNR, spectral flatness, clipping %

6. **Interview Persona** ([interviewer/](backend/ai_agent/interviewer/))
   - **Controller** ([controller.py](backend/ai_agent/interviewer/controller.py)): Session lifecycle
   - **Evaluator** ([evaluator.py](backend/ai_agent/interviewer/evaluator.py)): Lenient scoring (50%+ coverage = correct)
   - Q-randomization: `hash(room_id + stage)` seed per-session
   - Scoring: 9 pts (correct first) → 7 (partial→correct) → 5 (partial→fail) → 4 (incorrect→correct) → 1 (fail)
   - Max: 90 pts (10 Qs: 4 basic + 3 moderate + 3 advanced)

7. **Persona Management** ([ai_personas/builder.py](backend/ai_personas/builder.py))
   - JSON config + presets
   - **Mandatory**: `metadata.source_template_id`
   - Voice settings per persona

### Service Boundaries

- **WebSocket signaling**: `/ws/signaling/{room_id}/` (Django Channels → [CallConsumer](backend/ai_agent/consumers.py))
- **Media streams**: aiortc (separate from signaling; peer connection)
- **Message flow**: `offer` → `answer` → `ice-candidate` + `transcript`, `agent_state`, `speaking_state`
- **Frontend hook**: [useWebRTC.ts](frontend/app/hooks/useWebRTC.ts) (peer connection + signaling management)
- **Auth endpoints**: `/` (login), `/dashboard` (persona selector), `/dashboard/personas` (designer UI)

---

## Non-Negotiable Patterns

### 1. Async/ORM Boundary
**Rule**: Never call Django ORM inside aiortc/Channels async context.
```python
# ✅ Correct
from channels.db import database_sync_to_async

@database_sync_to_async
def get_persona_config(room_id):
    return Room.objects.select_related('call__persona').get(id=room_id)

config = await get_persona_config(room_id)

# ❌ Wrong — deadlock/crash
config = Room.objects.get(id=room_id)  # Inside async coroutine
```

### 2. Utterance ID Binding
**Rule**: Bind `utterance_id` when audio capture starts; reuse after STT.
```python
# Capture start
utterance_id = str(uuid.uuid4())
self.active_utterance_id = utterance_id

# After STT completes
await conversation.process_user_utterance(
    utterance_id=self.active_utterance_id,  # Same ID
    transcript=text
)
```
**Why**: Prevents race conditions where barge-in/state changes invalidate mid-processing utterances.

### 3. Audio Format (Strict)
**Rule**: All audio paths → 48kHz s16 mono.
```python
SAMPLE_RATE = 48000
FRAME_DURATION = 0.02  # 20ms
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION)  # 960
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # 1920 bytes
```
If input differs: resample immediately using `librosa` or `scipy.signal.resample`.

### 4. Turn-Based STT Only
**Rule**: No partial/streaming transcripts — one complete STT call per turn.
```python
# ✅ Correct: Buffer until 2.5s silence, then single STT
if self.silence_frame_count >= SILENCE_FRAME_COUNT:
    buffer_bytes = b''.join(self.audio_buffer)
    text = await self.stt_provider.transcribe(buffer_bytes)

# ❌ Wrong: Fragments user turn
if len(self.audio_buffer) > threshold:
    partial = await self.stt_provider.transcribe(...)
```

### 5. Barge-In Activation Window
**Rule**: Barge-in detection only active during `ConversationPhase.AI_SPEAKING`.
```python
if conversation.phase == ConversationPhase.AI_SPEAKING:
    barge_in_sm.on_audio_frame(frame_bytes)
    if barge_in_sm.state == BargeinState.ACCEPTED:
        await tts_track.cancel_playback()
```

---

## Debugging & Common Issues

### Log Markers (Quick Reference)
- 🔊 VAD/turn decisions
- ✅ Successful operations  
- ❌ Errors/failures
- 🛑 Critical stops (barge-in, turn abort)
- 📊 Audio metrics (SNR, energy, spectral flatness)

### Troubleshooting
1. **Empty transcripts** → Low SNR buffer (check audio_conditioning.py logs)
2. **No audio playback** → FFmpeg missing or Coqui TTS not installed
3. **WebSocket crashes** → Using `runserver` instead of `daphne`
4. **API key errors** → Missing `GROQ_API_KEY`/`SARVAM_API_KEY` with `AI_MODE=live`
5. **Low interview scores** → Check evaluator thresholds in [evaluator.py](backend/ai_agent/interviewer/evaluator.py) (50%+ coverage = correct)
6. **Same questions repeatedly** → Verify randomization seed in [controller.py](backend/ai_agent/interviewer/controller.py#L273)

### Migrations
```bash
cd backend
python manage.py makemigrations
python manage.py migrate
```
Remember: Still wrap ORM in `database_sync_to_async` in async code paths.

### Audio Recordings & Artifacts
- Per-room recordings: [backend/test_recordings/{room_id}/](backend/test_recordings/)
- Includes: raw input, conditioned output, utterance segments

---

## Essential Documentation

- [VAD_ARCHITECTURE_FIX.md](VAD_ARCHITECTURE_FIX.md) - Turn-taking rules, 2.5s silence threshold rationale
- [AUDIO_CONDITIONING_README.md](AUDIO_CONDITIONING_README.md) - 8-step pipeline details, metrics
- [INTELLIGENT_BARGE_IN_README.md](INTELLIGENT_BARGE_IN_README.md) - Persona-aware interruption logic
- [STOP_WORD_IMPLEMENTATION_COMPLETE.md](STOP_WORD_IMPLEMENTATION_COMPLETE.md) - Mid-TTS interruption patterns
- [README.md](README.md) - Setup quickstart, architecture overview
