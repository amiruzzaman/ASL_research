import sys
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from models.utils import extract_landmarks


def frame_capture(path):
    cap = cv.VideoCapture(path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))



def get_data(batch_size = 64):
    dataset = fouh.load_from_hub("Voxel51/WLASL")
    session = fo.launch_app(dataset)

if __name__ == "__main__":
    get_data()