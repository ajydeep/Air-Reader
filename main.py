import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyautogui
import time

model = joblib.load("pre/best_gesture_model.pkl")


viewer_path = "viewer.html"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Starting gesture detection... Press 'q' to quit.")

last_action_time = 0
action_cooldown = 0.5  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            data = np.array([[thumb_tip.x, thumb_tip.y, index_tip.x, index_tip.y]])
            prediction = model.predict(data)[0]

            current_time = time.time()
            if current_time - last_action_time > action_cooldown:
                if prediction == 1:
                    print("Gesture: Zoom In")
                    pyautogui.hotkey('command', '+')
                    last_action_time = current_time
                elif prediction == 2:
                    print("Gesture: Zoom Out")
                    pyautogui.hotkey('command', '-')
                    last_action_time = current_time
                else:
                    print("Gesture: None")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
