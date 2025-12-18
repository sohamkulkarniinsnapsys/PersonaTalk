from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Room, Call

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']


class RoomSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    persona = serializers.SerializerMethodField()

    class Meta:
        model = Room
        fields = ['id', 'owner', 'name', 'created_at', 'is_active', 'persona']

    def get_persona(self, obj):
        # Get the most recent Call with a persona for this room
        call = obj.calls.filter(persona__isnull=False).order_by('-started_at').first()
        if call and call.persona:
            return {
                'slug': call.persona.slug,
                'display_name': call.persona.display_name
            }
        return None

class CallSerializer(serializers.ModelSerializer):
    class Meta:
        model = Call
        fields = '__all__'
