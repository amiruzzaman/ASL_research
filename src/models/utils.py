import torch


def create_mask(src, trg, pad_idx, device):
    # Get sequence length
    src_seq_len = src.shape[1]
    trg_seq_len = trg.shape[1]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(trg_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = src == pad_idx
    tgt_padding_mask = trg == pad_idx
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_square_subsequent_mask(size, device):
    mask = torch.tril(torch.ones((size, size), device=device)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, -torch.inf)
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def extract_landmarks(landmarks):
    return torch.tensor(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    ).flatten()


def get_feature(results):
    left_hand = (
        extract_landmarks(results.left_hand_landmarks)
        if results.left_hand_landmarks
        else torch.zeros(63)
    )
    right_hand = (
        extract_landmarks(results.right_hand_landmarks)
        if results.right_hand_landmarks
        else torch.zeros(63)
    )
    pose = (
        extract_landmarks(results.pose_landmarks)
        if results.pose_landmarks
        else torch.zeros(99)
    )
    feature = torch.cat((left_hand, right_hand, pose))

    return feature
