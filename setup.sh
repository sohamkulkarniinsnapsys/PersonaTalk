#!/bin/bash

echo "Setting up Video Conference App Environment..."

# Backend Setup
echo "Checking Backend..."
mkdir -p backend

# Venv Setup
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment found."
fi

# Activate venv and install requirements
echo "Activating venv and installing requirements..."
source venv/bin/activate

if [ -f "backend/requirements.txt" ]; then
    pip install -r backend/requirements.txt
else
    echo "No requirements.txt found yet. Skipping pip install."
fi

# Frontend Setup
echo "Checking Frontend..."
if [ ! -d "frontend" ]; then
    echo "Frontend directory not found. Please run 'npx create-next-app@latest frontend --typescript --app' manually or let the agent do it."
else
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo "Setup complete!"
