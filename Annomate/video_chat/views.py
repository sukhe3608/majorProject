from django.shortcuts import render

def video_chat_room(request, room_name):
    return render(request, '/video_call.html', {'room_name': room_name})
