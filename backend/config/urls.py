"""
URL configuration for config project.
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from api.views import RegisterView

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Auth Routes
    path('', auth_views.LoginView.as_view(template_name='registration/login.html', redirect_authenticated_user=True), name='login'),
    path('accounts/register/', RegisterView.as_view(), name='register'),
    path('accounts/logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    
    # Standard APIs
    path('api/', include('ai_personas.urls')),
    path('api/', include('api.urls')),
    path('api-auth/', include('rest_framework.urls')), 
]
