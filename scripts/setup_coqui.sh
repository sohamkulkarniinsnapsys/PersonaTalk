#!/bin/bash
# setup_coqui.sh

echo "Setting up Coqui TTS environment..."

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it (brew install ffmpeg / sudo apt install ffmpeg)"
    exit 1
fi

cd "$(dirname "$0")/../backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt
pip install TTS numpy soundfile pydub ffmpeg-python

echo "Pre-downloading default model..."
python3 -c "from TTS.api import TTS; TTS('tts_models/en/vctk/vits')"

echo "Setup complete!"
