import streamlit as st
import cv2
import mediapipe as mp
import os
from datetime import datetime
import pyrebase
import threading

from dotenv import load_dotenv
load_dotenv()

firebaseConfig = {
    "apiKey": st.secrets["firebase"]["FIREBASE_API_KEY"],
    "authDomain": st.secrets["firebase"]["FIREBASE_AUTH_DOMAIN"],
    "projectId": st.secrets["firebase"]["FIREBASE_PROJECT_ID"],
    "storageBucket": st.secrets["firebase"]["FIREBASE_STORAGE_BUCKET"],
    "messagingSenderId": st.secrets["firebase"]["FIREBASE_MESSAGING_SENDER_ID"],
    "appId": st.secrets["firebase"]["FIREBASE_APP_ID"],
    "databaseURL": st.secrets["firebase"]["FIREBASE_DATABASE_URL"]
}
# Initialize global variables
lat, long = 22.294858, 73.362279
hand_gestures = {}

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Firebase configuration


firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()

# Directory to save images
output_directory = 'captured_images'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Reset gesture state function
def reset_gesture_state(hand_id):
    if hand_id in hand_gestures:
        hand_gestures[hand_id]['state'] = 0
        hand_gestures[hand_id]['k'] = 0
        if hand_gestures[hand_id]['timer']:
            hand_gestures[hand_id]['timer'].cancel()
            hand_gestures[hand_id]['timer'] = None
        print(f"Gesture reset due to timeout for hand {hand_id}")

# Detect gestures and upload function
def detect_gesture_and_upload():
    global hand_gestures
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)

        if res.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract finger tip positions
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                middle_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                ring_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                pinky_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

                # Initialize hand state
                if idx not in hand_gestures:
                    hand_gestures[idx] = {'state': 0, 'k': 0, 'timer': None}

                # Gesture detection logic
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
                    img_filename = f'image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                    img_path = os.path.join(output_directory, img_filename)
                    cv2.imwrite(img_path, frame)
                    print(f'Saved {img_filename}')

                    cloudpath = f"images/{img_filename}"
                    storage.child(cloudpath).put(img_path)
                    print(f"Image {img_filename} successfully uploaded to Firebase.")

                    db.child("location").set({"latitude": lat, "longitude": long})
                    print(f"Location {lat}, {long} successfully sent to Firebase.")
                    if hand_gestures[idx]['timer']:
                        hand_gestures[idx]['timer'].cancel()
                    hand_gestures[idx]['timer'] = threading.Timer(3.0, reset_gesture_state, args=(idx,))
                    hand_gestures[idx]['timer'].start()

        # Display video feed
        cv2.imshow('Gesture Detection', frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Guardian Gesture",
        page_icon="🤝",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Header Section
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            color: #2b7a78;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #3aafa9;
            text-align: center;
        }
        .button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.9rem;
            color: #555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">Guardian Gesture</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-enabled Gesture Recognition System</div>', unsafe_allow_html=True)

    # Display an image or icon
    st.image(
        "https://via.placeholder.com/600x300.png?text=Guardian+Gesture+App",
        caption="Real-time Gesture Recognition and Alert System",
        use_container_width=True,
    )

    # Start gesture detection
    st.markdown('<div class="button">', unsafe_allow_html=True)
    if st.button("🖐 Start Gesture Detection"):
        threading.Thread(target=detect_gesture_and_upload, daemon=True).start()
        st.success("Gesture detection has started! Please use the camera to detect gestures.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add some instructions for users
    st.markdown(
        """
        ### How to Use:
        1. Ensure your camera is connected and working.
        2. Click the "Start Gesture Detection" button.
        3. Perform the predefined gestures to trigger the system.
        4. Press `q` in the video feed window to quit.
        """,
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown('<div class="footer">Powered by AI and Firebase | © 2025 Guardian Gesture</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
