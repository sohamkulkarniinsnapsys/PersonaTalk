from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.views.generic import CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme
from django.conf import settings
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
import json

class RegisterView(CreateView):
    form_class = UserCreationForm
    template_name = 'registration/register.html'
    success_url = reverse_lazy('login')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, "Account created successfully! Please sign in.")
        
        # Optional: Auto-login
        # user = form.save()
        # login(self.request, user)
        # return redirect(self.get_success_url())
        
        return response

    def get_success_url(self):
        # Securely handle 'next' parameter if we were to auto-login or redirect elsewhere
        next_url = self.request.GET.get('next')
        if next_url and url_has_allowed_host_and_scheme(
            url=next_url,
            allowed_hosts={self.request.get_host()},
            require_https=self.request.is_secure()
        ):
            return next_url
        return super().get_success_url()


@require_http_methods(["GET"])
def csrf_token_view(request):
    """Provide CSRF token for frontend"""
    return JsonResponse({'csrfToken': get_token(request)})


@require_http_methods(["GET"])
def auth_check_view(request):
    """Check if user is authenticated"""
    return JsonResponse({
        'authenticated': request.user.is_authenticated,
        'username': request.user.username if request.user.is_authenticated else None
    })


@require_http_methods(["POST"])
@csrf_protect
def login_api(request):
    """JSON API endpoint for login with field-specific error handling"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        logger.info(f"Login attempt for username: {username}")
        
        # Validate required fields
        if not username or not password:
            errors = {}
            if not username:
                errors['username'] = 'Username is required'
            if not password:
                errors['password'] = 'Password is required'
            
            return JsonResponse({
                'success': False,
                'errors': errors
            }, status=400)
        
        # Authenticate against database
        user = authenticate(request, username=username, password=password)
        
        logger.info(f"Authentication result for {username}: {'Success' if user else 'Failed'}")
        
        if user is not None:
            # Successful authentication
            login(request, user)
            logger.info(f"User {username} logged in successfully")
            return JsonResponse({
                'success': True,
                'redirectUrl': '/dashboard',
                'username': user.username
            })
        else:
            # Authentication failed - provide specific error
            # Check if username exists to give helpful feedback
            user_exists = User.objects.filter(username=username).exists()
            
            logger.info(f"User exists check for {username}: {user_exists}")
            
            if user_exists:
                # Username exists but password is wrong
                return JsonResponse({
                    'success': False,
                    'errors': {
                        'password': 'Incorrect password'
                    }
                }, status=401)
            else:
                # Username doesn't exist
                return JsonResponse({
                    'success': False,
                    'errors': {
                        'username': 'Username not found'
                    }
                }, status=401)
                
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid request format'
        }, status=400)
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred during login'
        }, status=500)


@require_http_methods(["POST"])
@csrf_protect
def register_api(request):
    """JSON API endpoint for registration"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()
        password1 = data.get('password1', '')
        password2 = data.get('password2', '')
        
        logger.info(f"Registration attempt for username: {username}")
        
        # Validate required fields
        errors = {}
        if not username:
            errors['username'] = 'Username is required'
        if not password1:
            errors['password'] = 'Password is required'
        if not password2:
            errors['password2'] = 'Please confirm your password'
            
        if errors:
            logger.warning(f"Registration validation failed for {username}: {errors}")
            return JsonResponse({
                'success': False,
                'errors': errors
            }, status=400)
        
        # Check if passwords match
        if password1 != password2:
            logger.warning(f"Password mismatch for {username}")
            return JsonResponse({
                'success': False,
                'errors': {
                    'password2': 'Passwords do not match'
                }
            }, status=400)
        
        # Check if username already exists
        if User.objects.filter(username=username).exists():
            logger.warning(f"Username {username} already exists")
            return JsonResponse({
                'success': False,
                'errors': {
                    'username': 'This username is already taken'
                }
            }, status=400)
        
        # Validate password strength (Django's built-in validators)
        from django.contrib.auth.password_validation import validate_password
        from django.core.exceptions import ValidationError as DjangoValidationError
        
        try:
            validate_password(password1, user=None)
        except DjangoValidationError as e:
            logger.warning(f"Password validation failed for {username}: {e.messages}")
            return JsonResponse({
                'success': False,
                'errors': {
                    'password': '; '.join(e.messages)
                }
            }, status=400)
        
        # Create the user
        user = User.objects.create_user(
            username=username,
            password=password1
        )
        
        logger.info(f"User {username} registered successfully")
        
        # Auto-login the user
        login(request, user)
        
        return JsonResponse({
            'success': True,
            'redirectUrl': '/dashboard',
            'username': user.username
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid request format'
        }, status=400)
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred during registration'
        }, status=500)
