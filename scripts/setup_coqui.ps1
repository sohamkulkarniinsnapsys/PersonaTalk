# setup_coqui.ps1
# PowerShell script to set up Coqui TTS for local development

Write-Host "Setting up Coqui TTS environment..." -ForegroundColor Cyan

# Check for ffmpeg
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "ffmpeg is present." -ForegroundColor Green
} else {
    Write-Host "ffmpeg not found! Please install ffmpeg and add it to your PATH." -ForegroundColor Red
    Write-Host "You can likely install it via 'winget install ffmpeg' or download from ffmpeg.org."
}

# Python environment
$venvPath = "..\backend\venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating python virtual environment..."
    python -m venv $venvPath
}

# Activate
& "$venvPath\Scripts\Activate.ps1"

Write-Host "Installing dependencies..."
# We need basic backend reqs + TTS
pip install -r ..\backend\requirements.txt
pip install TTS numpy soundfile pydub ffmpeg-python

Write-Host "Dependencies installed."

# Optional: Download model
Write-Host "The Coqui model will be downloaded automatically on first run."
Write-Host "To pre-download, you can run a simple python script:"
$downloadScript = @"
try:
    from TTS.api import TTS
    print('Downloading default model tts_models/en/vctk/vits...')
    TTS('tts_models/en/vctk/vits')
    print('Model downloaded.')
except Exception as e:
    print(f'Error: {e}')
"@

# Run the snippet
python -c $downloadScript

Write-Host "Setup complete. Set AI_MODE=mock and run your backend." -ForegroundColor Green
