# Setup Script for One-on-One AI Video Conference
Write-Host "Setting up AI Video Conference Environment..."

# Backend Setup
Write-Host "Setting up Backend..."
cd backend
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Created virtual environment."
}
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate

# Create default persona if not exists
python manage.py shell -c "from ai_personas.models import Persona; Persona.objects.get_or_create(slug='default', defaults={'display_name': 'Default AI', 'config': {'system_prompt': 'You are a helpful assistant.'}})"

cd ..

# Frontend Setup
Write-Host "Setting up Frontend..."
cd frontend
npm install
cd ..

Write-Host "Setup Complete!"
Write-Host "To run backend: cd backend; .\venv\Scripts\Activate; daphne -p 8000 config.asgi:application"
Write-Host "To run frontend: cd frontend; npm run dev"
Write-Host "Visit http://localhost:3000/dashboard/personas to use the Persona Designer."
Write-Host "Set AI_MODE='live' and provide env vars for real providers."
