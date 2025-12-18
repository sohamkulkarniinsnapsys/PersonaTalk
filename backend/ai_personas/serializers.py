from rest_framework import serializers
from .models import Persona, PersonaVersion
import jsonschema
from .builder import PERSONA_SCHEMA
import logging

logger = logging.getLogger(__name__)

class PersonaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Persona
        fields = [
            'uuid', 'slug', 'display_name', 'description_text',
            'config', 'is_active', 'version', 'created_at', 'updated_at'
        ]
        read_only_fields = ['uuid', 'created_at', 'updated_at', 'version']

    def validate_config(self, value):
        try:
            logger.info(f"Validating config: {value}")
            jsonschema.validate(instance=value, schema=PERSONA_SCHEMA)
        except jsonschema.ValidationError as e:
            logger.error(f"Config validation failed: {e.message}")
            logger.error(f"Failed config: {value}")
            raise serializers.ValidationError(f"Invalid config: {e.message}")
        return value

class PersonaVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonaVersion
        fields = '__all__'

class ConvertPersonaSerializer(serializers.Serializer):
    description_text = serializers.CharField()
    voice_hint = serializers.CharField(required=False, allow_blank=True)
    template_id = serializers.CharField(required=False, allow_null=True)
    current_voice = serializers.DictField(required=False)
    preserve_fields = serializers.ListField(child=serializers.CharField(), required=False)

class PreviewPersonaSerializer(serializers.Serializer):
    config = serializers.JSONField()
    text = serializers.CharField(required=False, default="Hello, can you hear me?")
