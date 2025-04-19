from django.urls import re_path
from .consumers import VideoChatConsumer

websocket_urlpatterns = [
    re_path(r'ws/video_chat/(?P<room_name>\w+)/$', VideoChatConsumer.as_asgi()),
]