import sys

import time
from dataprocess import get_data
from models.glosstoenglish.model import GlossToEnglishModel
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm 

from models.utils import create_mask
from models.utils import generate_square_subsequent_mask

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def inference(args):
    _, _, gloss_vocab, gloss_id, text_vocab, text_id = get_data(args.batch)
    model = GlossToEnglishModel(len(gloss_vocab), len(text_vocab), args.dmodel, args.heads, args.encoders, args.decoders).to(DEVICE)

    # If the save data argument is not null, then we load
    if args.model_path:
        print("Loading model...")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    while True:
        print("~~ Translate ASL Gloss to English Sentence ~~")
        sequence = input("ASL Sequence: ")

        if sequence.lower() == "<stop>":
            break
        
        # Turns the string input into a tensor containing tokens 
        tokens = torch.tensor([gloss_vocab[word if word in gloss_vocab else "<unk>"] for word in sequence.split()])
        num_tokens = tokens.shape[0]
        mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)

        # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string 
        translated_tokens = model.greedy_decode(tokens, mask, gloss_vocab, text_vocab, device=DEVICE).flatten()
        translated = " ".join([text_id[token] for token in translated_tokens.tolist()]).replace("<sos>", "").replace("<eos>", "")
        print(f"Translated Sequence: {translated[1:]}\n")

    
def train_epoch(model, data, optimizer, criterion, src_vocab, trg_vocab):
    # Set model to training mode 
    model.train()
    
    # Go through batches in the epoch
    for src, trg in tqdm(data):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Create the masks for the source and target
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, trg_vocab["<pad>"], DEVICE)

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model 
        # instead of the model's output in the prior timestep
        out = model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)

        # For the criterion function to work, we have to concatenate all the batches together for it to work
        # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size) 
        # to (Batch * Sequence Size, Target Vocab Size)
        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)
        
        # We zero the gradients of the model, calculate the total loss of the sample
        # Then compute the gradient vector for the model over the loss

        # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
        # because it allows us to compare the next word the model predicts to the actual next word in the sequence  
        loss = criterion(actual, expected)
        loss.backward()

        # Apply the gradient vector on the trainable parameters in the model and reset the gradients
        optimizer.step()
        optimizer.zero_grad()

def validate(model, data, criterion, src_vocab, trg_vocab):
    losses = 0
    model.eval()
    # Go through batches in the epoch
    for src, trg in tqdm(data):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Create the masks for the source and target
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, trg_vocab["<pad>"], DEVICE)

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model 
        # instead of the model's output in the prior timestep
        out = model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)
        
        # For the criterion function to work, we have to concatenate all the batches together for it to work
        # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size) 
        # to (Batch * Sequence Size, Target Vocab Size)
        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)
        
        # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
        # because it allows us to compare the next word the model predicts to the actual next word in the sequence  
        loss = criterion(actual, expected)
        losses += loss

    # Returns the average loss
    return losses / len(list(data))


def train(args):
    train_dl, test_dl, gloss_vocab, gloss_id, text_vocab, text_id = get_data(args.batch)
    
    # Creating the translation (Transformer) model
    EPOCHS = args.epochs
    curr_epoch = 1
    model = Translator(len(gloss_vocab), len(text_vocab), args.dmodel, args.heads, args.encoders, args.decoders).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    best_loss = -torch.inf

    # If the save data argument is not null, then we load the data to continue training
    if args.model_path:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.model_path, weights_only=False)

        curr_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        best_loss = checkpoint['best_loss']

    for epoch in range(curr_epoch, EPOCHS + 1):
        # Train through the entire training dataset and keep track of total time
        start_time = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, criterion, gloss_vocab, text_vocab)
        total_time = time.time() - start_time

        # Goes through the validation dataset 
        valid_loss = validate(model, test_dl, criterion, gloss_vocab, text_vocab)

        # If the average loss from testing the validation data is smallest than the best model at that point,
        # Then we save the current model 
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("New best model, saving...")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion,
                    'best_loss': best_loss
                    }, args.save_path + "best.pt")

        # Saves model's data after current epoch (Checkpoint system)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion,
                    'best_loss': best_loss
                    }, args.save_path + f"epoch_{epoch}.pt")

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Valid Loss: {valid_loss:.3f}")
        print(f"Epoch Time: {total_time:.1f} seconds")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ASLGlossModel")
    
    # Training or inference mode
    parser.add_argument('--train', action='store_true')
    
    # Training procedure
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('-b', '--batch', type=int, default=32)
        
    # Translation Model Arguments
    parser.add_argument('--dmodel', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--encoders', type=int, default=2)
    parser.add_argument('--decoders', type=int, default=2)

    parser.add_argument('--save_path', type=str, default=f"./")
    args = parser.parse_args()
    
    # Either train the model or use the model
    if args.train:
        train(args)
    else:
        inference(args)


