# llm_api/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'^ws/interview/(?P<session_id>[0-9a-fA-F\-]+)/$', consumers.InterviewConsumer.as_asgi()),
]