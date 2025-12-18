"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from django.core.asgi import get_asgi_application

# Load env before Django settings (critical for Daphne startup)
here = Path(__file__).resolve().parent
backend_dir = here.parent
repo_root = backend_dir.parent
load_dotenv(dotenv_path=backend_dir / ".env", override=False)
load_dotenv(dotenv_path=repo_root / ".env", override=False)
load_dotenv(find_dotenv(usecwd=True), override=False)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import ai_agent.routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            ai_agent.routing.websocket_urlpatterns
        )
    ),
})
