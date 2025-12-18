import pytest
import shutil
from django.test import TestCase
from ai_personas.models import Persona
from ai_personas.serializers import PersonaSerializer
from ai_agent.utils import build_prompt
from ai_agent.ai_orchestrator import AIOrchestrator
from ai_agent.providers import MockSTT, MockLLM, MockTTS

class PersonaModelTests(TestCase):
    def test_create_persona_valid(self):
        config = {
            "system_prompt": "Test Prompt",
            "voice": {"provider": "mock"}
        }
        persona = Persona.objects.create(
            slug="test-persona",
            display_name="Test",
            config=config
        )
        self.assertEqual(persona.slug, "test-persona")
    
    def test_config_validation(self):
        # Should raise error if system_prompt is missing validation in model save? 
        # Note: Model validation triggers on clean(), save() doesn't always call it unless overriden.
        # But our validator was added to the field. Django doesn't enforce field validators on save() automatically.
        # Explicit clean call for test:
        config = {"invalid": "config"}
        persona = Persona(slug="bad", display_name="Bad", config=config)
        with self.assertRaises(Exception): # ValidationError
            persona.full_clean()

class BuildPromptTests(TestCase):
    def test_build_prompt_structure(self):
        config = {
            "system_prompt": "System",
            "examples": [{"user": "Hi", "assistant": "Ho"}]
        }
        history = [{"role": "user", "content": "Prev"}]
        transcript = "New input"
        
        messages = build_prompt(config, history, transcript)
        
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn("System", messages[0]['content'])
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], 'Hi')
        self.assertEqual(messages[2]['role'], 'assistant') # Example response
        self.assertEqual(messages[3]['role'], 'user') # History
        self.assertEqual(messages[3]['content'], 'Prev')
        self.assertEqual(messages[-1]['role'], 'user')
        self.assertEqual(messages[-1]['content'], 'New input')

@pytest.mark.asyncio
@pytest.mark.django_db
class TestOrchestrator:
    async def test_handle_utterance_mock_flow(self):
        # Setup DB (needs pytest-django transactional db access if we use DB)
        # For simplicity, if we cannot use async DB easily in this setup without plugins, 
        # we might need to mock get_object or use sync_to_async wrappers correctly.
        # But we added sync_to_async in orchestrator, so it should work if DB is accessible.
        
        # We need to create a persona first. Since this is async test, we need sync wrapper or async create.
        from asgiref.sync import sync_to_async
        
        await sync_to_async(Persona.objects.create)(
            slug="integration-test",
            display_name="Int Test",
            config={"system_prompt": "You are a test.", "voice": {}}
        )
        
        orchestrator = AIOrchestrator()
        
        # Mock providers are default
        assert isinstance(orchestrator.stt, MockSTT)
        
        # Test handle_utterance
        # MockSTT returns "Hello there..." regardless of input
        room_id = "room-1"
        audio_bytes = b"fakeaudio"
        
        result = await orchestrator.handle_utterance(room_id, audio_bytes, "integration-test")
        
        assert result is not None
        assert result['transcript'] == "Hello there, how are you?" # MockSTT fixed response
        assert result['response_text'] != "" 
        assert result['tts_audio'] is not None
        assert len(result['tts_audio']) > 0
