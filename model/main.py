import time
from dataprocess import get_data
from glosstoenglish import Translator
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm 

# Train on the GPU if possible
PATH = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def beam_search():
    pass

def greedy_decode(src, src_mask, model, src_vocab, trg_vocab, max_len=10):
    # Convert the sequences from (Sequence) to (Batch, Sequence)
    src = src.unsqueeze(0).to(DEVICE)

    # Feed the source sequence and its mask into the transformer's encoder
    memory = model.encode(src, src_mask)

    # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
    sequence = torch.ones(1, 1).fill_(trg_vocab["<sos>"]).type(torch.long).to(DEVICE)

    for _ in range(max_len):
        mask = generate_square_subsequent_mask(sequence.shape[-1]).type(torch.bool).to(DEVICE)

        # Feeds the target and retrieves a vector (Batch, Sequence Size, Target Vocab Size)
        out = model.decode(sequence, memory, mask)
        next_word = torch.argmax(out[:, -1], dim=-1)

        # Concatenate the predicted token to the output sequence
        sequence = torch.cat([sequence, torch.ones(1, 1).fill_(next_word.item()).type(torch.long).to(DEVICE)], dim=1)

        if next_word == trg_vocab["<eos>"]:
            break
    
    return sequence

def inference(model, src_vocab, src_id, trg_vocab, trg_id):
    model.eval()
    
    while True:
        sequence = input()

        if sequence.lower() == "<stop>":
            break
        
        # Turns the string input into a tensor containing tokens 
        tokens = torch.tensor([src_vocab[word if word in src_vocab else "<unk>"] for word in sequence.split()])
        num_tokens = tokens.shape[0]
        mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)

        # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string 
        translated_tokens = greedy_decode(tokens, mask, model, src_vocab, trg_vocab).flatten()
        translated = " ".join([trg_id[token] for token in translated_tokens.tolist()]).replace("<sos>", "").replace("<eos>", "")

    
def train(model, data, optimizer, criterion, src_vocab, trg_vocab):
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
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, trg_vocab["<pad>"])

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
        optimizer.zero_grad()
        loss = criterion(actual, expected)
        loss.backward()

        # Apply the gradient vector on the trainable parameters in the model 
        optimizer.step()

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
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, trg_vocab["<pad>"])

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


def create_mask(src, trg, pad_idx):
    # Get sequence length
    src_seq_len = src.shape[1]
    tgt_seq_len = trg.shape[1]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (trg == pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1)
    mask = mask.float().masked_fill(mask == 0, -torch.inf).masked_fill(mask == 1, float(0.0))
    return mask

def main():
    train_dl, test_dl, gloss_vocab, gloss_id, text_vocab, text_id = get_data(64)
    
    # Creating the translation (Transformer) model
    EPOCHS = 10
    model = Translator(len(gloss_vocab), len(text_vocab), 512, 8, 2, 2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    best_loss = -torch.inf
    for epoch in range(1, EPOCHS + 1):
        # Train through the entire training dataset and keep track of total time
        start_time = time.time()
        train(model, train_dl, optimizer, criterion, gloss_vocab, text_vocab)
        total_time = time.time() - start_time

        # Goes through the validation dataset 
        valid_loss = validate(model, test_dl, criterion, gloss_vocab, text_vocab)

        # If the average loss from testing the validation data is smallest than the best model at that point,
        # Then we save the current model 
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("New best model, saving...")
            torch.save(model.state_dict(), PATH + "best.pt")

        # Saves model's data after current epoch (Checkpoint system)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion
                    }, PATH + f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()


