import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from models.signtogloss.datasetloader import load_data
from models.signtogloss.model import SignToGlossModel
from models.utils import extract_landmarks


if __name__ == "__main__":
    # cap = cv.VideoCapture("data/WLASL/start_kit/raw_videos/12328.mp4")
    cap = cv.VideoCapture("C:/Users/RedAP/Pictures/Camera Roll/WIN_20250329_15_41_30_Pro.mp4")
    "C:/Users/RedAP/Pictures/Camera Roll/WIN_20250329_15_41_30_Pro.mp4"
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes, classes, gloss_to_id, _, _ = load_data()
    id_to_gloss = {id:gloss for gloss, id in gloss_to_id.items()}

    model = SignToGlossModel(225, 50, 512, device=DEVICE)
    checkpoint = torch.load("C:/Users/RedAP/Desktop/test_data/epoch_1000.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    features = []
    while True:
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
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,
        #     mp_holistic.FACEMESH_CONTOURS,
        #     mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
        #     mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1)
        # )

        left_hand = extract_landmarks(results.left_hand_landmarks) if results.left_hand_landmarks else torch.zeros(63)
        right_hand = extract_landmarks(results.right_hand_landmarks) if results.right_hand_landmarks else torch.zeros(63)
        pose = extract_landmarks(results.pose_landmarks) if results.pose_landmarks else torch.zeros(99)
        feature = torch.cat((left_hand, right_hand, pose))
        features.append(feature)
        
        # # Drawing Right hand Land Marks
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # # Drawing Left hand Land Marks
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # cv.imshow('frame', image)
        # keyCode = cv.waitKey(1)
        # if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
        #     break
    
    temp = torch.tensor([[feature.tolist() for feature in features]])
    out = model(temp, [temp.size(dim=1)], device=DEVICE)
    out = out[:, -1]
    print(out[0].argmax(dim=-1))
    print(id_to_gloss[out[0].argmax(dim=-1).item()])
    
    cap.release()
    # cv.destroyAllWindows()