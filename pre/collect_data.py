import cv2
import mediapipe as mp
import numpy as np
import csv
import math
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

csv_file = open('../data/gesture_data.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['thumb_x', 'thumb_y', 'index_x', 'index_y', 'label']) 

last_save_time = time.time()
COOLDOWN = 0.2  

print("Instructions:")
print("1.  'i' to save ZOOM IN (fingers closing)")
print("2.  'o' to save ZOOM OUT (fingers opening)")
print("3.  'n' to save NO ACTION (static)")
print("4.  'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, "Press 'i' (zoom in), 'o' (zoom out), 'n' (no action)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Collect Data', frame)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key in [ord('i'), ord('o'), ord('n')] and (time.time() - last_save_time) > COOLDOWN:
        if results.multi_hand_landmarks:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            if key == ord('i'):
                label = 1  
                print("Zoom In")
            elif key == ord('o'):
                label = 2  
                print("Zoom Out")
            elif key == ord('n'):
                label = 0  
                print("No Action")
            
            writer.writerow([thumb.x, thumb.y, index.x, index.y, label])
            last_save_time = time.time()

csv_file.close()
cap.release()
cv2.destroyAllWindows()


