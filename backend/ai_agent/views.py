import os
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from .models import Room, Call
from .serializers import RoomSerializer, CallSerializer


class RoomViewSet(viewsets.ModelViewSet):
    queryset = Room.objects.all()
    serializer_class = RoomSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        # Attach persona if provided in payload (expects slug)
        persona_slug = self.request.data.get('persona')
        room = serializer.save(owner=self.request.user)
        if persona_slug:
            # Create initial Call object linking persona to this room
            from ai_personas.models import Persona
            try:
                persona = Persona.objects.get(slug=persona_slug)
                Call.objects.create(room=room, persona=persona)
            except Persona.DoesNotExist:
                # Invalid persona provided; log and fallback to default
                import logging
                logging.getLogger(__name__).warning(f"Room created with invalid persona slug: {persona_slug}")
                # Still create a default call so that webrtc.py can find a persona
                try:
                    default_persona = Persona.objects.get(slug='default')
                    Call.objects.create(room=room, persona=default_persona)
                except Persona.DoesNotExist:
                    pass

    def create(self, request, *args, **kwargs):
        # Validate persona slug if present before creating room
        persona_slug = request.data.get('persona')
        if persona_slug:
            from ai_personas.models import Persona
            if not Persona.objects.filter(slug=persona_slug).exists():
                return Response({'error': 'Invalid persona provided'}, status=400)
        return super().create(request, *args, **kwargs)

    def get_queryset(self):
        # Users see their own rooms
        return self.queryset.filter(owner=self.request.user)

@api_view(['GET'])
@permission_classes([permissions.AllowAny]) # Changing to AllowAny for easier local dev, typically IsAuthenticated
def get_ice_config(request):
    """
    Returns STUN/TURN configuration.
    For local dev, returns public Google STUN servers.
    """
    # In production, fetch TURN credentials (e.g., from Coturn REST API or env vars)
    ice_servers = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
    ]
    
    # Example logic for TURN if env vars present
    turn_server = os.getenv('TURN_SERVER')
    turn_user = os.getenv('TURN_USER')
    turn_pass = os.getenv('TURN_PASS')
    
    if turn_server and turn_user and turn_pass:
        ice_servers.append({
            "urls": turn_server,
            "username": turn_user,
            "credential": turn_pass
        })

    return Response({"iceServers": ice_servers})
