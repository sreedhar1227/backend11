# llm_interview_system/asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import llm_api.routing  # We'll create this file next

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'llm_interview_system.settings')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket': AuthMiddlewareStack(
        URLRouter(
            llm_api.routing.websocket_urlpatterns
        )
    ),
})