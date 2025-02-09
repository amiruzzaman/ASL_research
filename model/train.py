from dataprocess import get_data
from glosstoenglish import Translator

import torch
import torch.nn as nn
import torch.optim as optim

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beam_search():
    pass

def greedy_decode(src, src_mask, model, src_vocab, trg_vocab, max_len=1000):
    # Convert the sequences from (Sequence) to (Batch, Sequence)
    src = src.unsqueeze(0).to(DEVICE)
    src_mask = src.unsqueeze(0).to(DEVICE)

    # Feed the source sequence and its mask into the transformer's encoder
    memory = model.encode(src, src_mask)

    # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
    sequence = torch.ones(1, 1).fill_(trg_vocab["<sos>"]).type(torch.long).to(DEVICE)

    for _ in range(max_len):
        mask = generate_square_subsequent_mask(sequence.shape[-1])

        # Feeds the target and retrieves a vector (Batch, Sequence Size, Target Vocab Size)
        out = model.decode(sequence, mask, memory)

        predicted = torch.argmax(out, dim=2)
        next_word = predicted[-1]

        # Concatenate the predicted token to the output sequence
        sequence = torch.cat([sequence, torch.ones(1, 1).fill_(next_word).type(torch.long).to(DEVICE)], dim=1)

        if next_word == trg_vocab["<eos>"]:
            break

    return sequence


def train(model, data, optimizer, criterion, src_vocab, trg_vocab):
    # Set model to training mode 
    model.train(mode=True)
    
    # Go through batches in the epoch
    for src, trg in data:
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Create the masks for the source and target
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg, trg_vocab["<pad>"])

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model 
        # instead of the model's output in the prior timestep
        out = model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask)

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


def create_mask(src, trg, pad_idx):
    # Get sequence length
    src_seq_len = src.shape[1]
    tgt_seq_len = trg.shape[1]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
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
    data, gloss_vocab, gloss_id, text_vocab, text_id = get_data(64)

    # Creating the translation (Transformer) model
    model = Translator(len(gloss_vocab), len(text_vocab), 512, 8, 2, 2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

if __name__ == "__main__":
    main()

