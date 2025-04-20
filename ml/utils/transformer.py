import torch

def convert_to_tokens(sequence, gloss_vocab, device):
    tokens = torch.tensor([gloss_vocab[word if word in gloss_vocab else "<unk>"] for word in sequence.split()]).to(device)
    tokens = torch.cat([torch.tensor([gloss_vocab["<sos>"]]).to(device), tokens, torch.tensor([gloss_vocab["<eos>"]]).to(device)]).to(device)
    return tokens

def create_mask(src, trg, pad_idx, device):
    # Get sequence length
    src_seq_len = src.shape[1]
    trg_seq_len = trg.shape[1]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(trg_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    
    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (trg == pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  

def generate_square_subsequent_mask(size, device):
    mask = (torch.tril(torch.ones((size, size), device=device)) == 1)
    mask = mask.float().masked_fill(mask == 0, -torch.inf).masked_fill(mask == 1, float(0.0))
    return mask
