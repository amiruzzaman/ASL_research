import os
import sys

import time
from models.signtogloss.datasetloader import load_data
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm 

from models.signtogloss.model import SignToGlossModel
from models.utils import create_mask
from models.utils import generate_square_subsequent_mask

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1
    
def train_epoch(model, data, optimizer, criterion, epoch, gloss_to_id):
    # Set model to training mode 
    model.train()
    losses, correct = 0, 0
    
    # Go through batches in the epoch
    for glosses, landmarks, seq_len in tqdm(data, desc= f"Epoch {epoch}"):
        landmarks = landmarks.to(DEVICE)
        ids = torch.tensor([gloss_to_id[gloss] for gloss in glosses]).to(DEVICE)

        out = model(landmarks, seq_len, device=DEVICE)
        out = out[range(out.shape[0]), torch.tensor(seq_len).to(DEVICE) - 1, :]

        loss = criterion(out, ids)
        losses += loss
        correct += (out.argmax(dim=-1) == ids).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    losses /= len(data)
    correct /= len(data.dataset)
    return losses, correct 

def validate(model, data, criterion, gloss_to_id):
    losses, correct = 0, 0
    model.eval()

    # Go through batches in the epoch
    for glosses, landmarks, seq_len in tqdm(data, desc= "Evaluating"):
        landmarks = landmarks.to(DEVICE)
        ids = torch.tensor([gloss_to_id[gloss] for gloss in glosses]).to(DEVICE)

        out = model(landmarks, seq_len, device=DEVICE)
        out = out[range(out.shape[0]), torch.tensor(seq_len).to(DEVICE) - 1, :]
    
        loss = criterion(out, ids)
        losses += loss
        correct += (out.argmax(dim=-1) == ids).sum().item()

    losses /= len(data)
    correct /= len(data.dataset)
    return losses, correct 

def train(args):
    num_classes, classes, gloss_to_id, train, test = load_data(args.batch)
    
    # Creating the translation (Transformer) model
    EPOCHS = args.epochs
    curr_epoch = 1
    model = SignToGlossModel(225, num_classes, 512, device=DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=args.adams_ep)
    best_loss = torch.inf

    # If the save data argument is not null, then we load the data to continue training
    if args.model_path:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.model_path, weights_only=False)

        curr_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        best_loss = checkpoint['best_loss']

    # Calculate starting performance of the model
    valid_loss, correct = validate(model, test, criterion, gloss_to_id)
    print(f"Starting Performance: \nValid Accuracy: {(100*correct):>0.1f}%, Valid Average loss: {valid_loss:>8f}\n")

    for epoch in range(curr_epoch, EPOCHS + 1):
        # Train through the entire training dataset and keep track of total time
        start_time = time.time()
        train_loss, train_correct = train_epoch(model, train, optimizer, criterion, epoch, gloss_to_id)
        
        # Goes through the validation dataset 
        valid_loss, correct = validate(model, test, criterion, gloss_to_id)

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
                    }, args.save_path + "/best.pt")

        # Saves model's data after current epoch (Checkpoint system)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion,
                    'best_loss': best_loss
                    }, args.save_path + f"/epoch_{epoch}.pt")

        total_time = time.time() - start_time

        print(f"\nEpoch Time: {total_time:.1f} seconds")
        print(f"Training Accuracy: {(100*train_correct):>0.1f}%, Training Average loss: {train_loss:>8f}")
        print(f"Valid Accuracy: {(100*correct):>0.1f}%, Valid Average loss: {valid_loss:>8f}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ASLGlossModel")
    
    # Training or inference mode
    parser.add_argument('--train', action='store_true')
    
    # Training procedure
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('-b', '--batch', type=int, default=10)
    parser.add_argument('--adams_ep', type=float, default=5e-9)

    parser.add_argument('--save_path', type=str, default=f"./")
    args = parser.parse_args()
    
    # Either train the model or use the model
    if args.train:
        train(args)


