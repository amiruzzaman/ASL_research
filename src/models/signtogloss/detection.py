import cv2 as cv
import mediapipe as mp
import torch
from models.signtogloss.datasetloader import load_data
from models.signtogloss.model import SignToGlossModel


from models.utils import get_feature

# Thirty videos worth of data
no_sequences = 50

# Videos are going to be 30 frames in length
sequence_length = 30

cap = cv.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.25, min_tracking_confidence=0.25
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes, _, _, id_to_gloss, gloss_to_id = load_data()
model = SignToGlossModel(225, num_classes, 512, device=DEVICE)
checkpoint = torch.load("C:/Users/RedAP/Desktop/test_data/best.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

while True:
    stop_running = False
    features = None

    for frame_num in range(sequence_length):
        # Read feed
        ret, frame = cap.read()

        if not ret:
            break

        image = frame

        results = holistic_model.process(image)
        landmarks = get_feature(results)
        features = (
            torch.cat((features, landmarks.view(1, -1)), dim=0)
            if features is not None
            else landmarks.view(1, -1)
        )

        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

        cv.imshow("OpenCV Feed", image)
        # Break gracefully
        if cv.waitKey(10) & 0xFF == ord("q"):
            stop_running = True
            break

    if stop_running:
        break

    word = ""
    if features is not None:
        id = torch.argmax(
            model(features.view(1, sequence_length, -1).to(DEVICE), device=DEVICE),
            dim=-1,
        )
        word = id_to_gloss[id.item()]

    cv.putText(
        image, word, (120, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv.LINE_AA
    )
    cv.imshow("OpenCV Feed", image)
    cv.waitKey(2000)

cap.release()
cv.destroyAllWindows()
