from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.core.cache import cache
from .models import Persona, PersonaVersion
from .serializers import (
    PersonaSerializer, PersonaVersionSerializer, 
    ConvertPersonaSerializer, PreviewPersonaSerializer
)
from .builder import PersonaBuilder
from .tts_providers import ProviderFactory
from asgiref.sync import async_to_sync
from django.http import HttpResponse, HttpResponseForbidden
import time
import logging

logger = logging.getLogger(__name__)

class PersonaViewSet(viewsets.ModelViewSet):
    queryset = Persona.objects.all()
    serializer_class = PersonaSerializer
    lookup_field = 'slug'

    def perform_create(self, serializer):
        # Initial version 1
        instance = serializer.save(version=1)
        # Create history
        PersonaVersion.objects.create(
            persona=instance,
            version=1,
            config=instance.config,
            description_text=instance.description_text
        )

    def perform_update(self, serializer):
        # Increment version
        instance = serializer.save()
        instance.version += 1
        instance.save()
        
        # Create history
        PersonaVersion.objects.create(
            persona=instance,
            version=instance.version,
            config=instance.config,
            description_text=instance.description_text
        )
        
        # Invalidate cache
        cache.delete(f"persona_config_{instance.slug}")

    @action(detail=True, methods=['get'])
    def config(self, request, slug=None):
        """
        Internal/Admin endpoint to get resolved config.
        """
        # TODO: Add authentication guard here (e.g. check for internal header or admin)
        cache_key = f"persona_config_{slug}"
        cached = cache.get(cache_key)
        if cached:
            return Response(cached)

        persona = self.get_object()
        config = persona.upgraded_config  # Use auto-upgraded config
        config['_meta'] = {
            'uuid': str(persona.uuid),
            'slug': persona.slug,
            'version': persona.version
        }
        cache.set(cache_key, config, timeout=60)
        return Response(config)

    @action(detail=False, methods=['post'])
    def convert(self, request):
        """
        Convert description -> config
        """
        # Rate Limiting for generation
        user_key = f"convert_limit_{request.user.id or request.META.get('REMOTE_ADDR')}"
        if cache.get(user_key):
             return Response({"error": "Rate limit exceeded. Please wait."}, status=429)
        cache.set(user_key, True, timeout=5)

        serializer = ConvertPersonaSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        desc = serializer.validated_data['description_text']
        current_voice = serializer.validated_data.get('current_voice')
        template_id = serializer.validated_data.get('template_id')
        
        builder = PersonaBuilder()
        
        try:
            # Async generation in sync view
            # Returns tuple: (config, raw_text, tokens)
            config, raw_text, tokens = async_to_sync(builder.a_generate_config)(
                desc, 
                current_voice=current_voice,
                template_id=template_id
            )
            
            return Response({
                "persona_config": config,
                "raw_llm_output": raw_text,
                "tokens": tokens
            })
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return Response({
                "error": str(e),
                "details": "LLM generation failed or produced invalid JSON."
            }, status=422)

    @action(detail=True, methods=['post'])
    def preview(self, request, slug=None):
        """
        Generate TTS preview for the persona's voice configuration.
        Expects JSON: { "text": "optional override", "voice_config": {...override...} }
        """
        # Rate limiting: Prevent concurrent synthesis requests (XTTS v2 is resource-intensive)
        user_ip = request.META.get('REMOTE_ADDR')
        limit_key = f"tts_preview_limit_{user_ip}"
        if cache.get(limit_key):
            return Response({
                "error": "A synthesis is already in progress. Please wait for it to complete.",
                "retry_after": 30
            }, status=status.HTTP_429_TOO_MANY_REQUESTS)
        
        # Lock for 30s (XTTS v2 can take 30-60s on first synthesis)
        cache.set(limit_key, True, timeout=30)
        
        text = request.data.get('text', 'This is a preview of my voice.')
        
        # Use persona upgraded config by default, allow override from request for live tweaking
        persona = self.get_object()
        voice_config = persona.upgraded_config.get('voice', {})
        
        # Merge overrides (shallow merge)
        if 'voice_config' in request.data:
            voice_config.update(request.data['voice_config'])
            
        try:
            tts = ProviderFactory.get_tts_provider()
            audio_bytes = async_to_sync(tts.synthesize)(text, voice_config)
            
            # Convert raw PCM to WAV with headers for browser compatibility
            import io
            import wave
            
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)      # Mono
                    wav_file.setsampwidth(2)      # 16-bit
                    wav_file.setframerate(48000)  # 48kHz (normalized in provider)
                    wav_file.writeframes(audio_bytes)
                
                final_wav_data = wav_buffer.getvalue()
            
            return HttpResponse(final_wav_data, content_type="audio/wav")
        except Exception as e:
            logger.error(f"Preview TTS failed: {e}")
            return Response({"error": str(e)}, status=500)

    @action(detail=True, methods=['post'])
    def test_tts(self, request, slug=None):
        """
        Legacy/Test Synthesize endpoint.
        """
        text = request.data.get('text', 'Hello test.')
        persona = self.get_object()
        voice_config = persona.upgraded_config.get('voice', {})
        
        # Merge overrides from request
        if 'voice_config' in request.data:
            voice_config.update(request.data['voice_config'])
        
        tts = ProviderFactory.get_tts_provider()
        
        try:
            audio_bytes = async_to_sync(tts.synthesize)(text, voice_config)
            
            # Convert raw PCM to WAV with headers for browser compatibility
            import io
            import wave
            
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)      # Mono
                    wav_file.setsampwidth(2)      # 16-bit
                    wav_file.setframerate(48000)  # 48kHz (normalized in provider)
                    wav_file.writeframes(audio_bytes)
                
                final_wav_data = wav_buffer.getvalue()

            return HttpResponse(final_wav_data, content_type="audio/wav")
        except Exception as e:
            return Response({"error": str(e)}, status=500)
