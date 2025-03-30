import json
import cv2 as cv
import mediapipe as mp
import os
import msgpack
import torch
from models.utils import extract_landmarks, get_feature

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('data', 'signs') 

with open(os.path.join(DATA_PATH, "labels.txt"), 'r') as file:
    actions = file.read().strip().split(",")

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

for action in actions: 
    try:
        action_dir = os.path.join(DATA_PATH, action)
        os.mkdir(action_dir)
    except:
        pass

cap = cv.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for action in actions:
    for sequence in range(no_sequences):
        features = None

        for frame_num in range(sequence_length):
            # Read feed
            ret, frame = cap.read()

            if not ret:
                break
            
            image = frame
            
            results = holistic_model.process(image)
            landmarks = get_feature(results)
            features = torch.cat((features, landmarks.view(1, -1)), dim=0) if features != None else landmarks.view(1, -1)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            

            if frame_num == 0: 
                cv.putText(image, 'STARTING COLLECTION', (120,200), \
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), \
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    # Show to screen
                cv.imshow('OpenCV Feed', image)
                cv.waitKey(2000)

            else: 
                cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), \
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                # Show to screen
                cv.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        
        print(features.shape)
        if features == None:
            continue

        with open(os.path.join(DATA_PATH, action, f"{sequence}.json"), 'w') as f:
            packed = json.dumps(features.tolist())
            f.write(packed)

            
                         
cap.release()
cv.destroyAllWindows()