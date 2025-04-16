import sys

import time
from ml.datasets.alsg import load_alsg_dataset
from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm 

from ml.tools.utils import convert_to_tokens, create_mask
from ml.tools.utils import generate_square_subsequent_mask

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

def inference(args):
    _, _, gloss_vocab, gloss_id, text_vocab, text_id = load_alsg_dataset(args.batch)
    model = TranslatorModel(len(gloss_vocab), len(text_vocab), args.dmodel, args.heads, args.encoders, args.decoders, args.dropout, device=DEVICE).to(DEVICE)
    
    # If the save data argument is not null, then we load
    if args.model_path:
        print("Loading model...")
        checkpoint = torch.load(args.model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    while True:
        print("~~ Translate ASL Gloss to English Sentence ~~")
        sequence = input("ASL Sequence: ")
    
        if sequence.lower() == "<stop>":
            break
        
        # Turns the string input into a tensor containing tokens 
        tokens = convert_to_tokens(sequence, gloss_vocab, DEVICE)
        
        num_tokens = tokens.shape[0]
        mask = torch.zeros(num_tokens, num_tokens).type(torch.bool).to(DEVICE)
        
        # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string 
        translated_tokens = None
        if args.greedy:
            translated_tokens = model.greedy_decode(tokens, mask, gloss_vocab, text_vocab, device=DEVICE).flatten()
        else:
            translated_tokens = model.beam_search(tokens, mask, gloss_vocab, text_vocab, device=DEVICE, beam_size=args.beam_size, max_len=num_tokens + 5, temperature=0.5).flatten()
        
        translated = " ".join([text_id[token] for token in translated_tokens.tolist()]).replace("<sos>", "").replace("<eos>", "")
        print(f"Translated Sequence: {translated[1:]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ASLGlossModel")

    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('--adams_ep', type=float, default=1e-9)
    parser.add_argument('--factor', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-9)
    
    # Translation Model Arguments
    parser.add_argument('--dmodel', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--encoders', type=int, default=2)
    parser.add_argument('--decoders', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--beam_size', type=int, default=25)
    parser.add_argument('--save_path', type=str, default=f"./")

    inference(parser.parse_args())