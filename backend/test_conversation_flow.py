#!/usr/bin/env python
"""
Quick test to verify conversation flow without WebRTC.
Tests: .env loading, MockSTT variation, conversation controller logic.
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load .env 
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

# Now import our modules
from ai_agent.conversation import ConversationSession, ConversationController
from ai_agent.ai_orchestrator import AIOrchestrator
from ai_agent.providers import MockSTT
from ai_personas.models import Persona

async def test_conversation_flow():
    print("=" * 70)
    print("🧪 CONVERSATION AI FLOW TEST")
    print("=" * 70)
    
    # 1. Test .env loading
    print("\n1️⃣  Testing .env file loading...")
    ai_mode = os.environ.get('AI_MODE', 'NOT_SET')
    print(f"   AI_MODE from environment: {ai_mode}")
    if ai_mode == 'live':
        print("   ✅ .env is being loaded by Django!")
    else:
        print(f"   ⚠️  AI_MODE={ai_mode} (expected 'live' if using .env)")
    
    # 2. Test MockSTT variation
    print("\n2️⃣  Testing MockSTT variation...")
    stt = MockSTT()
    responses = []
    for i in range(5):
        response = await stt.transcribe(b'test_audio')
        responses.append(response)
        print(f"   [{i+1}] STT: '{response}'")
    
    unique = len(set(responses))
    if unique > 1:
        print(f"   ✅ MockSTT returns varied responses ({unique} unique)!")
    else:
        print(f"   ❌ MockSTT returns same response every time")
    
    # 3. Load persona
    print("\n3️⃣  Testing persona configuration...")
    try:
        persona = Persona.objects.filter(display_name='test3').first()
        if persona:
            print(f"   ✅ Found persona: {persona.display_name}")
            print(f"   Slug: {persona.slug}")
            print(f"   Flow mode: {persona.config.get('flow', 'NOT_SET')}")
            if persona.config.get('flow') == 'interview':
                print("   ✅ Flow is set to 'interview' (correct for dynamic Q&A)")
            else:
                print(f"   ⚠️  Flow is {persona.config.get('flow')} (expected 'interview')")
        else:
            print("   ⚠️  test3 persona not found")
    except Exception as e:
        print(f"   ❌ Error loading persona: {e}")
    
    # 4. Test conversation controller initialization
    print("\n4️⃣  Testing conversation controller...")
    try:
        orchestrator = AIOrchestrator()
        session = ConversationSession('test_room', 'test3-mj72jmnt', persona.config)
        controller = ConversationController(session, orchestrator, lambda x: None)
        print(f"   ✅ Controller initialized for room 'test_room'")
        print(f"   Initial phase: {session.phase}")
        print(f"   Flow mode: {session.persona_config.get('flow', 'NOT_SET')}")
    except Exception as e:
        print(f"   ❌ Error initializing controller: {e}")
    
    # 5. Test VAD threshold
    print("\n5️⃣  Testing VAD threshold configuration...")
    vad_threshold = int(os.environ.get('VAD_THRESHOLD', '800'))
    print(f"   VAD threshold: {vad_threshold}")
    if vad_threshold >= 800:
        print("   ✅ VAD threshold is high enough to filter noise!")
    else:
        print(f"   ⚠️  VAD threshold {vad_threshold} is low (might catch garbage noise)")
    
    print("\n" + "=" * 70)
    print("✅ TESTS COMPLETE")
    print("=" * 70)
    print("\nNext: Start a call via browser and test the conversation flow!")
    print("Expected: STT varies, AI asks contextual questions, conversation continues")
    

if __name__ == '__main__':
    asyncio.run(test_conversation_flow())
