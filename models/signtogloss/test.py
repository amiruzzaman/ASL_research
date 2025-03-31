import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from models.utils import extract_landmarks


if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        frame = cv.resize(frame, (800, 600))
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        results = holistic_model.process(image)

        # Converting back the RGB image to BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Drawing Face landmarks
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1)
        )
            
        data = None
        if results.left_hand_landmarks and results.right_hand_landmarks and results.face_landmarks:
            data = torch.cat([extract_landmarks(results.left_hand_landmarks), 
                                extract_landmarks(results.right_hand_landmarks), 
                                extract_landmarks(results.face_landmarks)])
        
        # Drawing Right hand Land Marks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Drawing Left hand Land Marks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        cv.imshow('frame', image)
        keyCode = cv.waitKey(1)
        if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()