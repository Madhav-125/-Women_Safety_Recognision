from flask import Flask, request, jsonify
import mediapipe as mp
import os
from datetime import datetime
import pyrebase
import threading
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)

lat, long = 22.294858, 73.362279
hand_gestures = {}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, 
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Firebase configuration
firebaseConfig = {
    "apiKey": "AIzaSyDUdttY89xF6y62aXAxWS0NmsZ35kzJPtE",
    "authDomain": "guardian-gesture.firebaseapp.com",
    "projectId": "guardian-gesture",
    "storageBucket": "guardian-gesture.appspot.com",
    "messagingSenderId": "503802348568",
    "appId": "1:503802348568:web:e75129179e7735369aaf6d",
    "databaseURL": "https://guardian-gesture-default-rtdb.firebaseio.com"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()

def reset_gesture_state(hand_id):
    if hand_id in hand_gestures:
        hand_gestures[hand_id]['state'] = 0
        hand_gestures[hand_id]['k'] = 0
        if hand_gestures[hand_id]['timer']:
            hand_gestures[hand_id]['timer'].cancel()
            hand_gestures[hand_id]['timer'] = None
        print(f"Gesture reset due to timeout for hand {hand_id}")

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global hand_gestures
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Read image from request
    image_file = request.files['image']
    img = Image.open(BytesIO(image_file.read()))
    img = np.array(img)
    
    # Convert to RGB (MediaPipe expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    res = hands.process(img_rgb)
    
    response_data = {
        'hands_detected': False,
        'gesture_completed': False
    }
    
    if res.multi_hand_landmarks:
        response_data['hands_detected'] = True
        
        for idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            if idx not in hand_gestures:
                hand_gestures[idx] = {'state': 0, 'k': 0, 'timer': None}

            if (thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y) and hand_gestures[idx]['state'] == 0:
                hand_gestures[idx]['state'] = 1
                hand_gestures[idx]['k'] += 1
                print(f"Hand {idx} closed")
                if hand_gestures[idx]['timer']:
                    hand_gestures[idx]['timer'].cancel()
                hand_gestures[idx]['timer'] = threading.Timer(3.0, reset_gesture_state, args=(idx,))
                hand_gestures[idx]['timer'].start()

            if (thumb_y > index_y and thumb_y > middle_y and thumb_y > ring_y and thumb_y > pinky_y) and hand_gestures[idx]['state'] == 1:
                if hand_gestures[idx]['k'] == 2:
                    hand_gestures[idx]['state'] = 2
                else:
                    hand_gestures[idx]['state'] = 0
                print(f"Hand {idx} opened")
                if hand_gestures[idx]['timer']:
                    hand_gestures[idx]['timer'].cancel()
                hand_gestures[idx]['timer'] = threading.Timer(3.0, reset_gesture_state, args=(idx,))
                hand_gestures[idx]['timer'].start()

            if (thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y) and hand_gestures[idx]['state'] == 2:
                hand_gestures[idx]['state'] = 0
                hand_gestures[idx]['k'] = 0
                print(f"Gesture completed for hand {idx}")
                response_data['gesture_completed'] = True
                
                # Save and upload image
                img_filename = f'image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                img_path = os.path.join('captured_images', img_filename)
                cv2.imwrite(img_path, img)
                
                cloudpath = f"images/{img_filename}"
                storage.child(cloudpath).put(img_path)
                
                db.child("location").set({"latitude": lat, "longitude": long})
                
                if hand_gestures[idx]['timer']:
                    hand_gestures[idx]['timer'].cancel()
                hand_gestures[idx]['timer'] = threading.Timer(3.0, reset_gesture_state, args=(idx,))
                hand_gestures[idx]['timer'].start()
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
    app.run(host='0.0.0.0', port=5000)
