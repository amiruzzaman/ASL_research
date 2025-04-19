import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
from ml.models.asl_to_english_v1.sign_to_gloss.model import SignToGlossModel

from ml.dataset_loaders.alsg import load_alsg_dataset
from ml.dataset_loaders.sign import load_sign_dataset

from ml.tools.utils import extract_landmarks

import json
import cv2 as cv
import mediapipe as mp
import os
import msgpack
import torch
from ml.tools.utils import extract_landmarks, get_feature

# Videos are going to be 30 frames in length
sequence_length = 30

cap = cv.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

PATH = os.path.join('ml', 'saved_models', 'sign_to_gloss')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading datasets
num_classes, _, _, id_to_gloss, gloss_to_id = load_sign_dataset()
_, _, gloss_vocab, gloss_id, text_vocab, text_id = load_alsg_dataset()

# Creating Mediapipe Hollistic model
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.25,
    min_tracking_confidence=0.25
)

# Creating Sign to Gloss model
sign_to_gloss = SignToGlossModel(225, num_classes, 512, device=DEVICE)
checkpoint = torch.load(os.path.join(PATH, 'best.pt'), weights_only=False)
sign_to_gloss.load_state_dict(checkpoint['model_state_dict'])

# Creating Gloss To English model
gloss_to_english = TranslatorModel(len(gloss_vocab), len(text_vocab), 512, 8, 2, 2, 0.3, device=DEVICE)
# checkpoint = torch.load("C:/Users/RedAP/Desktop/save_data/best.pt", weights_only=False)
# gloss_to_english.load_state_dict(checkpoint['model_state_dict'])

is_running = True

def get_sequence():
    res = []

    while True:
        stop_running = False
        features = None
    
        for frame_num in range(sequence_length):
            # Read feed
            ret, image = cap.read()

            if not ret:
                break
                
            results = holistic_model.process(image)
            landmarks = get_feature(results)
            features = torch.cat((features, landmarks.view(1, -1)), dim=0) if features is not None else landmarks.view(1, -1)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            cv.imshow('OpenCV Feed', image)
            # Break gracefully
            if cv.waitKey(10) & 0xFF == ord('q'):
                quit()
                
        if stop_running:
            break
        
        word = ""
        if features is not None:
            sign_to_gloss.eval()
            id = torch.argmax(sign_to_gloss(features.view(1, sequence_length, -1).to(DEVICE), device=DEVICE), dim=-1)
            word = id_to_gloss[id.item()]
        
        res.append(word)
        cv.putText(image, ' '.join(res), (150,150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)     
        cv.imshow('OpenCV Feed', image)
        k = cv.waitKey(2000)
        
        if k == ord(' '):
            break
        elif k == ord('c'):
            res.clear()
    
    return ' '.join(res)


while is_running:
   sequence = get_sequence()
   print(sequence)
    



cap.release()
cv.destroyAllWindows()