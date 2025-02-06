import torch
import torch.nn as nn
from positionalencoding import PositionalEncoding

class Translator(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, heads, num_encoders=1, num_decoders=1, 
                 dropout=0.1, max_len=1000, activation="relu"):
        
        super(Translator, self).__init__()

        # Embedding layers for the source and target inputs
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # Positional encodings for the embedding vectors
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(d_model, heads, num_encoders, num_decoders)
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, src_mask, trg_mask):
        # Turn the list of tokens into its embedding vectors and then apply positional encoding to those vectors
        src_pos = self.pos_encoding(self.src_embedding(src))
        trg_pos = self.pos_encoding(self.trg_embedding(trg))

        out = self.transformer(src, trg, src_mask, trg_mask)
        
        # Get the probability vector
        return self.softmax(self.linear(out))