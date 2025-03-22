import json
import os
import sys
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from models.utils import extract_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(path, args):
    cap = cv.VideoCapture(path)
    features = []
    
    while True:
        # Read the current frame
        success, frame = cap.read()

        if not success:
            break
        
        # Resize based on console args
        if args.width and args.height:
            frame = frame.resize(args.width, args.height)
        
        # Feeds in frame into holistic model
        results = holistic_model.process(frame)

        # Extracts and concatenate (x, y, z) data from the left and right hand landmarks to create a 1d tensor
        data = torch.cat([extract_landmarks(results.left_hand_landmarks), extract_landmarks(results.right_hand_landmarks)])
        features.append(data)

    return features

# Loading data json
with open("data/WLASL/start_kit/WLASL_v0.3.json", 'r') as file:
    dataset = json.load(file)

for label in dataset:
    gloss = label["gloss"]
    instances = label["instance"]
    print(f"Gloss: {gloss}")

    for instance in tqdm(instances, "Instances: "):
        id = instance["video_id"]
        source_path = os.path.join('data', 'WLASL', 'start_kit', 'videos', id + '.mp4')

        if not os.path.exists(source_path):
            continue
            
        features = extract_features(source_path)
    
