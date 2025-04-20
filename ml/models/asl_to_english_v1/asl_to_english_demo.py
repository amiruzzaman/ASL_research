import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.api.asl_to_english import ASLToEnglish
from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
from ml.models.asl_to_english_v1.sign_to_gloss.model import SignToGlossModel

from ml.dataloaders.alsg_dataloader import load_alsg_dataset
from ml.dataloaders.sign_dataloader import load_sign_dataset

from ml.utils.landmarks import extract_landmarks, get_feature, concatenate

import json
import cv2 as cv
import mediapipe as mp
import os
import msgpack
import torch


# Videos are going to be 30 frames in length
sequence_length = 30

cap = cv.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

PATH = os.path.join('ml', 'saved_models', 'sign_to_gloss')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating Mediapipe Hollistic model
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.25,
    min_tracking_confidence=0.25
)

model = ASLToEnglish()
is_running = True

def draw_connections(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
def get_sign():
    features = torch.empty(0)

    for frame_num in range(sequence_length):
        ret, image = cap.read()

        if not ret:
            break
        
        # Get landmark data from the current image and 
        results = holistic_model.process(image)
        landmarks = get_feature(results)
        features = concatenate(features, landmarks)
        
        # Draw the hand connections and show screen
        draw_connections(image, results)
        cv.imshow('OpenCV Feed', image)
        cv.waitKey(10)
    
    return features

while is_running:
   sequence = torch.empty(0)

   for i in range(5):
        sign = get_sign()
        ret, image = cap.read()

        # Adds the sign onto the sequence and translates it
        sequence = concatenate(sequence, sign)
        id, word = model.translate_sign(sign)

        cv.putText(image, word, (150,150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)     
        cv.imshow('OpenCV Feed', image)
        k = cv.waitKey(2000)

   print(model.translate(sequence))

cap.release()
cv.destroyAllWindows()