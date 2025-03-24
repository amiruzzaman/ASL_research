import json
import os
import sys
import warnings
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.utils import extract_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_data_from_video(path):
    cap = cv.VideoCapture(path)
    frames = []
    features = []
    
    while True:
        # Read the current frame
        success, frame = cap.read()
        
        if not success:
            break
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True
        
        # Extracts and concatenate (x, y, z) data from the left and right hand landmarks to create a 1d tensor
        left_hand = extract_landmarks(results.left_hand_landmarks) if results.left_hand_landmarks else torch.zeros(63)
        right_hand = extract_landmarks(results.right_hand_landmarks) if results.right_hand_landmarks else torch.zeros(63)
        pose = extract_landmarks(results.pose_landmarks) if results.pose_landmarks else torch.zeros(99)
        feature = torch.cat((left_hand, right_hand, pose))
        
        frames.append(image)
        features.append(feature)

    return frames, features

# Loading data json
with open("data/WLASL/start_kit/WLASL_v0.3.json", 'r') as file:
    dataset = json.load(file)

processed = []
for label in dataset[:5]:
    gloss = label["gloss"]
    instances = label["instances"]
    print(f"\nGloss: {gloss}")

    for instance in tqdm(instances, "Instances: "):
        id = instance["video_id"]
        source_path = os.path.join('data', 'WLASL', 'start_kit', 'raw_videos', id + '.mp4')
        
        if not os.path.exists(source_path):
            continue
        
        frames, features = extract_data_from_video(source_path)
        data = [feature.tolist() for feature in features]
        processed.append({"label": gloss, "features": data})

with open('wsasl.json', 'w') as f:
    json.dump(processed, f, indent=4)


            
        
        
        
