from django.http import JsonResponse
import cv2
import mediapipe as mp
import numpy as np
import threading
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django . shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User

def start_drawing():

    # Initialize Mediapipe and other settings
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    # Color and shape settings
    draw_color = (255, 0, 0)  # Initial color (Blue)
    brush_thickness = 5
    eraser_thickness = 50
    shape = "freehand"  # Default drawing mode
    drawing_area = (150, 150, 800, 600)  # Defined drawing area (x1, y1, x2, y2)
    is_filled = True  # Default to filled shapes
    
    # Shapes, colors, and buttons positions
    shapes_positions = {
        "rectangle": (50, 70),
        "circle": (50, 170),
        "freehand": (50, 270),
        "erase": (50, 370)  # Eraser option
    }
    colors_positions = {
        (760, 60): (255, 0, 0),   # Blue
        (760, 120): (0, 255, 0),   # Green
        (760, 180): (0, 0, 255),   # Red
        (760, 240): (255, 255, 255), # White
        (760, 300): (0, 255, 255), # Yellow
        (760, 360): (255, 165, 0), # Orange
        (760, 420): (128, 0, 128), # Purple
        (760, 480): (128, 128, 128) # Gray
    }
    toggle_fill_position = (50, 470)  # Position for toggling between filled/hollow
    clear_all_position = (50, 570)  # Position for the clear all button
    
    # Function to calculate distance between two points
    def distance(pt1, pt2):
        return int(np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2))
    
    # Function to check if the hand is in a closed fist gesture
    def is_closed_fist(landmarks, threshold=40):
        wrist = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        thumb_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
        index_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
        middle_tip = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        ring_tip = (int(landmarks[16].x * w), int(landmarks[16].y * h))
        pinky_tip = (int(landmarks[20].x * w), int(landmarks[20].y * h))
    
        distances = [
            distance(wrist, thumb_tip),
            distance(wrist, index_tip),
            distance(wrist, middle_tip),
            distance(wrist, ring_tip),
            distance(wrist, pinky_tip)
        ]
        
        return all(d < threshold for d in distances)
    
    cap = cv2.VideoCapture(0)
    canvas = None
    prev_index_finger_tip = None
    drawing_active = True  # Control whether drawing is active
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (800, 650))
    
        if canvas is None:
            canvas = np.zeros_like(frame)
    
        # Hand landmark detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
    
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_draw.DrawingSpec(color=(121,22,76),thickness = 2 , circle_radius=4),
                                      mp_draw.DrawingSpec(color=(125,44,250),thickness = 2 , circle_radius=2))
                landmarks = hand_landmarks.landmark
    
                h, w, _ = frame.shape
                index_finger_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
                thumb_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
    
                if is_closed_fist(landmarks):
                    drawing_active = False
                    prev_index_finger_tip = None
                else:
                    drawing_active = True
    
                if drawing_active:
                    # Toggle fill/hollow shapes
                    if abs(index_finger_tip[0] - toggle_fill_position[0]) < 30 and abs(index_finger_tip[1] - toggle_fill_position[1]) < 30:
                        is_filled = not is_filled
    
                    # Shape Selection Logic
                    for shape_name, shape_pos in shapes_positions.items():
                        if abs(index_finger_tip[0] - shape_pos[0]) < 30 and abs(index_finger_tip[1] - shape_pos[1]) < 30:
                            shape = shape_name
    
                    # Color Selection Logic
                    for color_pos, color in colors_positions.items():
                        if abs(index_finger_tip[0] - color_pos[0]) < 30 and abs(index_finger_tip[1] - color_pos[1]) < 30:
                            draw_color = color
    
                    # Clear All Logic
                    if abs(index_finger_tip[0] - clear_all_position[0]) < 30 and abs(index_finger_tip[1] - clear_all_position[1]) < 30:
                        canvas = np.zeros_like(frame)
    
                    # Drawing Logic (within the specified area)
                    if drawing_area[0] < index_finger_tip[0] < drawing_area[2] and drawing_area[1] < index_finger_tip[1] < drawing_area[3]:
                        shape_size = distance(index_finger_tip, thumb_tip)
                        if shape == "rectangle":
                            if is_filled:
                                cv2.rectangle(canvas, (index_finger_tip[0] - shape_size, index_finger_tip[1] - shape_size),
                                              (index_finger_tip[0] + shape_size, index_finger_tip[1] + shape_size), draw_color, cv2.FILLED)
                            else:
                                cv2.rectangle(canvas, (index_finger_tip[0] - shape_size, index_finger_tip[1] - shape_size),
                                              (index_finger_tip[0] + shape_size, index_finger_tip[1] + shape_size), draw_color, 3)
                        elif shape == "circle":
                            if is_filled:
                                cv2.circle(canvas, (index_finger_tip[0], index_finger_tip[1]), shape_size, draw_color, cv2.FILLED)
                            else:
                                cv2.circle(canvas, (index_finger_tip[0], index_finger_tip[1]), shape_size, draw_color, 3)
                        elif shape == "erase":
                            # Erase option
                            cv2.circle(canvas, index_finger_tip, eraser_thickness, (0, 0, 0), cv2.FILLED)
                        elif shape == "freehand":
                            if prev_index_finger_tip:
                                cv2.line(canvas, prev_index_finger_tip, index_finger_tip, draw_color, brush_thickness)
                        # Store the current tip position to connect the next point
                        prev_index_finger_tip = index_finger_tip
    
        # Draw the shapes menu, color palette, toggle option, and clear all button
        for shape_name, shape_pos in shapes_positions.items():
            color = (255, 0, 0) if shape == shape_name else (255, 255, 255)
            if shape_name == "rectangle":
                cv2.rectangle(frame, (shape_pos[0] - 25, shape_pos[1] - 25), (shape_pos[0] + 25, shape_pos[1] + 25), color, 3)
            elif shape_name == "circle":
                cv2.circle(frame, shape_pos, 25, color, 3)
            elif shape_name == "freehand":
                cv2.line(frame, (shape_pos[0] - 25, shape_pos[1]), (shape_pos[0] + 25, shape_pos[1]), color, 3)
            elif shape_name == "erase":
                cv2.rectangle(frame, (shape_pos[0] - 25, shape_pos[1] - 25), (shape_pos[0] + 25, shape_pos[1] + 25), color, 3)
                cv2.putText(frame, 'Erase', (shape_pos[0] - 50, shape_pos[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    
        for color_pos in colors_positions.keys():
            cv2.circle(frame, color_pos, 25, colors_positions[color_pos], cv2.FILLED)
    
        # Toggle fill/hollow option
        fill_color = (0, 255, 0) if is_filled else (0, 0, 255)
        cv2.rectangle(frame, (toggle_fill_position[0] - 25, toggle_fill_position[1] - 25),
                      (toggle_fill_position[0] + 25, toggle_fill_position[1] + 25), fill_color, 3)
        cv2.putText(frame, 'Fill' if is_filled else 'Hollow', (toggle_fill_position[0] - 25, toggle_fill_position[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_color, 2, cv2.LINE_AA)
    
        # Clear all button
        cv2.rectangle(frame, (clear_all_position[0] - 25, clear_all_position[1] - 25),
                      (clear_all_position[0] + 25, clear_all_position[1] + 25), (0, 0, 255), 3)
        cv2.putText(frame, 'Clear All', (clear_all_position[0] - 50, clear_all_position[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    
        # Combine the canvas and frame
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
        cv2.imshow("Virtual Canvas", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


def start_drawing_view(request):
    threading.Thread(target=start_drawing).start()
    return JsonResponse({'status': 'Drawing started'})


# def main(request):
#     return render(request,'main.html')

# @login_required(login_url='/login_page')
def home(request):
    return render(request, 'main.html')

@login_required(login_url='/login_page')
def index(request):
    if request.method=='POST':
        title = request.POST.get('title') 
        print(title)   
    return render(request,'index.html')

@login_required(login_url='/login_page')
def about(request):
    return render(request , 'about.html')

# def login_page(request):
#     if request.method == 'POST':
#         fnm = request.POST.get('fnm')
#         pwd = request.POST.get('pwd')
#         print(fnm , pwd)
#         userr = authenticate(request,username= fnm , password = pwd)
#         if userr is not None:
#             login(request,userr)
#             return redirect('/index')
#         else:
#             return redirect('login_page')
#     return render(request,'login_page.html')

from django.contrib.auth import authenticate, login
from django.shortcuts import redirect, render

def login_page(request):
    if request.method == 'POST':
        fnm = request.POST.get('fnm')
        pwd = request.POST.get('pwd')
        print(f"Received username: {fnm} and password: {pwd}")
        
        userr = authenticate(request, username=fnm, password=pwd)
        if userr is not None:
            login(request, userr)
            print(f"User {request.user} is authenticated: {request.user.is_authenticated}")  # Debug line
            return redirect('/index')
        else:
            print("Invalid credentials")
            return redirect('login_page')
    return render(request, 'login_page.html')


# @login_required(login_url='/login_page')
# def signup(request):
#     if request.method == 'POST':
#         fnm = request.POST.get('fnm')
#         emailid = request.POST.get('email')
#         pwd = request.POST.get('pwd')

#         print(fnm,emailid,pwd)
#         my_user = User.objects.create_user(fnm , emailid , pwd)
#         my_user.save()

#         return redirect('/login_page')
#     return render(request, 'signup.html')


def signup(request):
    if request.method == 'POST':
        fnm = request.POST.get('fnm')
        emailid = request.POST.get('email')
        pwd = request.POST.get('pwd')

        if not fnm or not emailid or not pwd:
            return render(request, 'signup.html', {'error_message': 'All fields are required.'})
        
        try:
            my_user = User.objects.create_user(fnm, emailid, pwd)
            my_user.save()
            return redirect('/login_page')
        except Exception as e:
            return render(request, 'signup.html', {'error_message': 'Signup failed. ' + str(e)})

    return render(request, 'signup.html')




# def btn_signup(request):
#     return render(request , '/signup')

# def btn_login(request):
#     return render(request , '/login_page')


@login_required(login_url='/login_page')
def chatroom(request):
    return render(request, 'chatroom.html')

