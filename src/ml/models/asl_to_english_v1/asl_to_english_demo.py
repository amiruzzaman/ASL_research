import cv2 as cv
import mediapipe as mp
import torch

from ml.api.asl_to_english import ASLToEnglish


import os


# Videos are going to be 30 frames in length
sequence_length = 30

cap = cv.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

PATH = os.path.join("ml", "saved_models", "sign_to_gloss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating Mediapipe Hollistic model
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.25, min_tracking_confidence=0.25
)

model = ASLToEnglish()
is_running = True


def draw_connections(image, results):
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )


def get_sign():
    buf = []

    for frame_num in range(sequence_length):
        ret, image = cap.read()

        if not ret:
            break

        buf.append(image)

        cv.imshow("OpenCV Feed", image)
        cv.waitKey(10)

    return buf


while is_running:
    sequence = []

    for i in range(5):
        sign = get_sign()
        ret, image = cap.read()

        # Adds the sign onto the sequence and translates it
        sequence.append(sign)
        id, word = model.translate_sign(sign)

        cv.putText(
            image,
            word,
            (150, 150),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            4,
            cv.LINE_AA,
        )
        cv.imshow("OpenCV Feed", image)
        k = cv.waitKey(2000)

    print(model.translate(sequence))

cap.release()
cv.destroyAllWindows()
