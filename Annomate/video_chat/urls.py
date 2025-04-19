from django.urls import path
from . import views

urlpatterns = [
    path('room/<str:room_name>/', views.video_chat_room, name='video_chat_room'),
]
