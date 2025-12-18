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
