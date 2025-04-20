import os
import torch
from ml.dataloaders.alsg_dataloader import load_alsg_dataset
from ml.dataloaders.sign_dataloader import load_sign_dataset
from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
from ml.models.asl_to_english_v1.sign_to_gloss.model import SignToGlossModel
from ml.tools.utils import convert_to_tokens

PATH = os.path.join('ml', 'saved_models')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ASLToEnglish:
    def __init__(self):
        # Loading datasets
        self.num_classes, _, _, self.id_to_gloss, _ = load_sign_dataset()
        _, _, self.gloss_vocab, self.gloss_to_id, self.text_vocab, self.text_to_id = load_alsg_dataset()

        # Creating Sign to Gloss model
        self.sign_to_gloss = SignToGlossModel(225, self.num_classes, 512, device=DEVICE)
        s2g_weights = torch.load(os.path.join(PATH, 'sign_to_gloss', 'best.pt'), weights_only=False)
        self.sign_to_gloss.load_state_dict(s2g_weights['model_state_dict'])
        
        # Creating Gloss To English model
        self.gloss_to_english = TranslatorModel(len(self.gloss_vocab), len(self.text_vocab), 512, 8, 2, 2, 0.3, device=DEVICE)
        g2e_weights = torch.load(os.path.join(PATH, 'gloss_to_english', 'best.pt'), weights_only=False)
        self.gloss_to_english.load_state_dict(g2e_weights['model_state_dict'])

        # Set both models to evaluation mode
        self.sign_to_gloss.eval()
        self.gloss_to_english.eval()

    def translate(self, signs):
        glosses = self.translate_signs(signs)
        sentence = self.translate_glosses(glosses)

        return sentence

    def translate_signs(self, sequence):
        sequence_length, _, _ = sequence.shape
        sequence = sequence.to(DEVICE)
        id = torch.argmax(self.sign_to_gloss(sequence, device=DEVICE), dim=-1)

        return ' '.join([self.id_to_gloss[id[i].item()] for i in range(sequence_length)])
    
    def translate_sign(self, sequence):
        sequence_length, _ = sequence.shape
        sequence = sequence.unsqueeze(dim=0).to(DEVICE)
        id = torch.argmax(self.sign_to_gloss(sequence, device=DEVICE), dim=-1)[0].item()

        return id, self.id_to_gloss[id]
    
    def translate_glosses(self, sequence):
        tokens = convert_to_tokens(sequence.split(), self.text_vocab)
        num_tokens = tokens.shape[0]
        mask = torch.zeros(num_tokens, num_tokens).type(torch.bool).to(DEVICE)
        
        # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string 
        translated_tokens = self.gloss_to_english.greedy_decode(tokens, mask, self.text_vocab, self.gloss_vocab, \
            device=DEVICE).flatten()
        
        translated = ' '.join([self.english_to_id[token] for token in translated_tokens.tolist()][1:-1])
        return translated
        