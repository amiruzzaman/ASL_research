import os

import torch
from ml.dataloaders.alsg_dataloader import load_alsg_dataset
from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
from ml.utils.transformer import convert_to_tokens

MODEL_PATH = os.getenv(
    "ENGLISH_2_GLOSS_MODEL_PATH", "./src/ml/saved_models/english_to_gloss/best.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnglishToGloss:
    def __init__(self):
        _, _, self.gloss_vocab, self.to_gloss, self.text_vocab, _ = load_alsg_dataset()
        self.model = TranslatorModel(
            len(self.text_vocab),
            len(self.gloss_vocab),
            512,
            8,
            2,
            2,
            0.3,
            device=DEVICE,
        )
        weights = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
        self.model.load_state_dict(weights["model_state_dict"])

        self.model.eval()
        self.invalid_words = {"be", "in", "to"}

    def translate(self, sequence):
        # Turns the string input into a tensor containing tokens
        tokens = convert_to_tokens(sequence.lower(), self.text_vocab, DEVICE)

        num_tokens = tokens.shape[0]
        mask = torch.zeros(num_tokens, num_tokens).type(torch.bool).to(DEVICE)

        # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string
        translated_tokens = self.model.greedy_decode(
            tokens, mask, self.text_vocab, self.gloss_vocab, device=DEVICE
        ).flatten()
        translated = [self.to_gloss[token] for token in translated_tokens.tolist()]
        translated = list(
            filter(lambda word: word not in self.invalid_words, translated)
        )

        return translated[1:-1]
