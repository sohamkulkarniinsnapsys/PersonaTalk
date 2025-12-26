"""
Update technical-interviewer persona with new question bank and config
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from ai_personas.models import Persona
from ai_personas.builder import PRESET_TEMPLATES
import copy

try:
    interviewer = Persona.objects.get(slug='technical-interviewer')
    print(f"\nFound persona: {interviewer.display_name}")
    print(f"Current total_questions: {interviewer.config.get('metadata', {}).get('interview_config', {}).get('total_questions', 'N/A')}")
    
    # Load fresh config from PRESET_TEMPLATES
    fresh_config = copy.deepcopy(PRESET_TEMPLATES['technical-interviewer'])
    
    # Update the persona config
    interviewer.config = fresh_config
    interviewer.save()
    
    # Verify update
    new_total = interviewer.config.get('metadata', {}).get('interview_config', {}).get('total_questions')
    basic_count = len(interviewer.config.get('metadata', {}).get('interview_config', {}).get('questions', {}).get('general', {}).get('basic', []))
    moderate_count = len(interviewer.config.get('metadata', {}).get('interview_config', {}).get('questions', {}).get('general', {}).get('moderate', []))
    advanced_count = len(interviewer.config.get('metadata', {}).get('interview_config', {}).get('questions', {}).get('general', {}).get('advanced', []))
    
    print(f"\n✅ Updated persona config!")
    print(f"New total_questions: {new_total}")
    print(f"General questions: {basic_count} basic, {moderate_count} moderate, {advanced_count} advanced")
    
    python_basic = len(interviewer.config.get('metadata', {}).get('interview_config', {}).get('questions', {}).get('python', {}).get('basic', []))
    python_moderate = len(interviewer.config.get('metadata', {}).get('interview_config', {}).get('questions', {}).get('python', {}).get('moderate', []))
    python_advanced = len(interviewer.config.get('metadata', {}).get('interview_config', {}).get('questions', {}).get('python', {}).get('advanced', []))
    print(f"Python questions: {python_basic} basic, {python_moderate} moderate, {python_advanced} advanced")
    
except Persona.DoesNotExist:
    print("\n❌ technical-interviewer persona not found!")
