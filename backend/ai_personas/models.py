import uuid
from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings

def validate_persona_config(value):
    pass # Deprecated, validation moved to serializer/builder


class Persona(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    slug = models.SlugField(max_length=100, unique=True)
    display_name = models.CharField(max_length=200)
    description_text = models.TextField(blank=True, help_text="Natural language description")
    config = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    version = models.IntegerField(default=1)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.display_name} (v{self.version})"

    @property
    def upgraded_config(self):
        """Auto-upgrade config with missing fields for backward compatibility."""
        config = self.config.copy() if self.config else {}
        
        # Ensure voice has model field (critical for XTTS v2)
        if 'voice' in config:
            if 'model' not in config['voice']:
                config['voice']['model'] = 'xtts_v2'
            if 'temperature' not in config['voice']:
                config['voice']['temperature'] = 0.75
        
        return config

    def save(self, *args, **kwargs):
        # Create audit version on save if it exists
        if self.pk:
            # Check if config changed? For now, we version every save or check diff.
            # Simple approach: Increment version and save history on every update that isn't just creation.
            # But we need the previous state.
            pass
        super().save(*args, **kwargs)

class PersonaVersion(models.Model):
    persona = models.ForeignKey(Persona, on_delete=models.CASCADE, related_name='versions')
    version = models.IntegerField()
    config = models.JSONField()
    description_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-version']
