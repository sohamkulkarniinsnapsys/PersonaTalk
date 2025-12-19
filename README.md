# One-on-One AI Video Conference

A one-on-one video conferencing web app using a Django backend and a modern React frontend (Next.js). The core UX: a single signed-in user opens a room, clicks Start Call, and immediately begins a live video + audio call with an AI persona.

## Prerequisites

- Python 3.8+
- Node.js 16+
- Docker & Docker Compose (optional but recommended)

## Setup

### Windows
```powershell
.\setup.ps1
```

### Linux/macOS
```bash
chmod +x setup.sh
./setup.sh
```

## Manual Setup

### Backend (Django)
```bash
# Create/Activate venv
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

cd backend
pip install -r requirements.txt
# For Coqui TTS support:
pip install TTS numpy soundfile pydub ffmpeg-python
# NOTE: You must also install FFmpeg on your system!
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

python manage.py migrate
# Create default persona
python manage.py shell -c "from ai_personas.models import Persona; Persona.objects.get_or_create(slug='default', defaults={'display_name': 'Default AI', 'config': {'system_prompt': 'You are a helpful assistant.'}})"
```

### Frontend (Next.js)
```bash
cd frontend
npm install
```

## Running the App

### Backend
```bash
cd backend
# Run with Mock AI (default)
daphne -p 8000 config.asgi:application

# Run with Live AI Providers (requires API keys)
# set keys in env variables or .env file
# export AI_MODE=live
# daphne -p 8000 config.asgi:application
```

### Frontend
```bash
cd frontend
npm run dev
```

Visit [http://localhost:3000/dashboard/personas](http://localhost:3000/dashboard/personas) to configure AI Personas.

## AI Configuration
The app uses a pluggable **Persona Layer** with professional-grade audio processing.

- **Admin UI**: Manage personas, prompts, and voices at `/dashboard/personas`.
- **Modes**: `AI_MODE=mock` (default) or `AI_MODE=live` (requires API keys).
- **Adapters**: STT, LLM, TTS are abstracted with mock/live implementations.
- **Audio Pipeline**: 8-step conditioning + turn-based interaction (see docs below)

### Voice Interaction Architecture
- ✅ Turn-based conversation (2.5s silence threshold - no mid-sentence cutoffs)
- ✅ Single STT call per complete turn (no fragmentation)
- ✅ Comprehensive finalization logging (explainable decisions)
- **Documentation**: [VAD Architecture Fix](VAD_ARCHITECTURE_FIX.md)

### Audio Conditioning Pipeline
- ✅ DC removal, noise suppression, AGC, speech-band emphasis
- ✅ Continuous noise floor learning (adapts to environment)
- ✅ Structured metrics logging (SNR, spectral flatness, clipping)
- **Documentation**: [Audio Conditioning Details](AUDIO_CONDITIONING_README.md)

### Testing Audio Pipeline
```bash
python test_audio_conditioning.py
```
