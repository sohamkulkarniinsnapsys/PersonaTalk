import uuid
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class Room(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='rooms')
    name = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Room {self.id} ({self.owner})"

class Call(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='calls')
    persona = models.ForeignKey('ai_personas.Persona', on_delete=models.SET_NULL, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    transcript = models.TextField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Call in {self.room.id} at {self.started_at}"


class InterviewSession(models.Model):
    """Persona-scoped persistent session for technical interviewer.

    STRICT ISOLATION: This model is used ONLY by the interviewer persona
    logic and is not referenced by other personas or shared flows.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    call = models.ForeignKey(Call, on_delete=models.CASCADE, related_name='interviewer_sessions')
    persona_slug = models.SlugField(max_length=100)
    total_score = models.FloatField(default=0.0)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["persona_slug"]),
        ]


class InterviewQuestionState(models.Model):
    """Persistent per-question lifecycle state for interviewer persona.

    Each question remains active until terminal resolution. Tracks both
    attempts, evaluation, hinting, and final scoring with derivation metadata.
    """
    class Evaluation(models.TextChoices):
        CORRECT = "correct", "correct"
        PARTIAL = "partial", "partial"
        INCORRECT = "incorrect", "incorrect"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE, related_name='questions')
    index = models.IntegerField(help_text="0-based order within interview")
    tech = models.CharField(max_length=64, default="general")
    stage = models.CharField(max_length=32, default="basic")
    question_text = models.TextField()
    reference_answer = models.TextField()
    key_concepts = models.JSONField(default=list, blank=True)

    first_response = models.TextField(blank=True)
    first_evaluation = models.CharField(max_length=16, choices=Evaluation.choices, blank=True)
    hint_given = models.BooleanField(default=False)
    second_response = models.TextField(blank=True)

    final_score = models.IntegerField(null=True, blank=True)
    finalized = models.BooleanField(default=False)
    feedback = models.TextField(blank=True)
    derivation = models.JSONField(default=dict, blank=True, help_text="Why this score was assigned")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("session", "index")
        ordering = ["index"]
