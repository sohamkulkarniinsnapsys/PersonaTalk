from django.urls import path, include
from rest_framework.routers import DefaultRouter
from ai_agent.views import RoomViewSet, get_ice_config
from . import views

router = DefaultRouter()
router.register(r'rooms', RoomViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('ice-config/', get_ice_config, name='ice-config'),
    path('csrf/', views.csrf_token_view, name='csrf'),
    path('auth/check/', views.auth_check_view, name='auth_check'),
    path('auth/login/', views.login_api, name='login_api'),
    path('auth/register/', views.register_api, name='register_api'),
]
