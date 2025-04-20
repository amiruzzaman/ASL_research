import torch

def extract_landmarks(landmarks):
    return torch.tensor([(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]).flatten()

def get_feature(results):
    left_hand = extract_landmarks(results.left_hand_landmarks) if results.left_hand_landmarks else torch.zeros(63)
    right_hand = extract_landmarks(results.right_hand_landmarks) if results.right_hand_landmarks else torch.zeros(63)
    pose = extract_landmarks(results.pose_landmarks) if results.pose_landmarks else torch.zeros(99)
    feature = torch.cat((left_hand, right_hand, pose))

    return feature

def concatenate(sequence, element):
    return torch.cat((sequence, element.unsqueeze(0)), dim=0) 