import asyncio
import pytest
from django.contrib.auth import get_user_model
from django.test import TestCase

from ai_agent.models import Room, Call, InterviewSession, InterviewQuestionState
from ai_personas.models import Persona
from ai_agent.interviewer.controller import InterviewerController
from ai_agent.interviewer.config import InterviewConfig


class DummyOrchestrator:
    class DummyTTS:
        async def synthesize(self, text, voice_cfg):
            return b"FAKE-PCM"
    def __init__(self):
        self.tts = self.DummyTTS()


async def _noop_send(audio, turn_id):
    return None


@pytest.mark.django_db
class InterviewerPersonaFlowTest(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create(username="u")
        self.room = Room.objects.create(owner=self.user, name="r1")
        
        # Create persona with dynamic config in metadata
        config_with_interview_settings = {
            "metadata": {
                "interview_config": {
                    "total_questions": 2,
                    "default_tech": "general",
                    "questions": {
                        "python": {
                            "basic": [
                                {
                                    "text": "What are lists vs tuples?",
                                    "answer": "Lists are mutable, tuples are immutable.",
                                    "concepts": ["mutable", "immutable"],
                                    "hint": "Think about whether they can change."
                                }
                            ]
                        }
                    }
                }
            },
            "voice": {}
        }
        
        self.persona = Persona.objects.create(
            slug="technical-interviewer",
            display_name="Interviewer",
            config=config_with_interview_settings
        )
        self.call = Call.objects.create(room=self.room, persona=self.persona)

    @pytest.mark.asyncio
    async def test_basic_lifecycle(self):
        ctl = InterviewerController(
            str(self.room.id), self.call.id, self.persona.slug, self.persona.config, DummyOrchestrator(), _noop_send
        )
        
        # Verify config loaded correctly
        assert ctl.config.total_questions == 2
        assert "python" in ctl.config.questions
        assert ctl.prompt_gen is not None
        
        await ctl.start()

        # First user input picks tech and asks first question
        await ctl.handle_user_utterance("Let's do Python")
        from asgiref.sync import sync_to_async
        sess = await ctl._get_or_create_session()
        q0 = await ctl._current_q(sess)
        assert q0 is not None
        assert q0.index == 0
        assert not q0.first_response

        # First attempt incorrect triggers hint, stays on same question
        await ctl.handle_user_utterance("I don't know")
        q0 = await ctl._current_q(sess)
        await sync_to_async(q0.refresh_from_db)()
        assert q0.first_response
        assert q0.first_evaluation == InterviewQuestionState.Evaluation.INCORRECT
        assert q0.hint_given is True

        # Second attempt finalizes and advances
        await ctl.handle_user_utterance("Lists are mutable and tuples are immutable")
        await sync_to_async(q0.refresh_from_db)()
        assert q0.finalized is True
        assert q0.final_score is not None
        
    @pytest.mark.asyncio
    async def test_dynamic_prompt_generation(self):
        """Test that prompt generator creates valid prompts from config."""
        ctl = InterviewerController(
            str(self.room.id), self.call.id, self.persona.slug, self.persona.config, DummyOrchestrator(), _noop_send
        )
        
        # Test greeting generation
        greeting = ctl.prompt_gen.generate_greeting()
        assert len(greeting) > 0
        assert "interview" in greeting.lower()
        
        # Test system prompt generation
        prompt = ctl.prompt_gen.generate_system_prompt()
        assert "ROLE AND PRIMARY OBJECTIVE" in prompt
        assert "QUESTION LIFECYCLE" in prompt
        assert "EVALUATION RULES" in prompt
        assert "HINTING RULES" in prompt
        assert "PROHIBITED BEHAVIORS" in prompt
        
        # Verify config values appear in prompt
        assert str(ctl.config.retry.max_attempts_per_question) in prompt
        assert str(int(ctl.config.scoring.concept_coverage_for_correct * 100)) in prompt
        
    @pytest.mark.asyncio
    async def test_final_evaluation_generation(self):
        """Test deterministic final evaluation feedback."""
        ctl = InterviewerController(
            str(self.room.id), self.call.id, self.persona.slug, self.persona.config, DummyOrchestrator(), _noop_send
        )
        
        # Test excellent score
        feedback = ctl.prompt_gen.generate_final_evaluation(
            total_score=18,
            max_score=20,
            weak_areas=[]
        )
        assert "90%" in feedback
        assert "very strong" in feedback.lower()
        
        # Test weak score with areas
        feedback = ctl.prompt_gen.generate_final_evaluation(
            total_score=5,
            max_score=20,
            weak_areas=["REST API", "Binary search", "HTTP codes"]
        )
        assert "25%" in feedback
        assert "weak" in feedback.lower()
        assert "REST API" in feedback
