import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_p=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = self.dropout(p=drop_p)
    
        # Create positional encoding matrix
        # max_len represents the maximum length a sequence of tokens can be
        # d_model represents the dimensions of the encoding vector
        self.pe = torch.zeros(max_len, d_model)   
        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        dim = torch.arange(0, d_model, 2)

        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / d_model)))
    
    def forward(self, x): 
        res = x + self.pe[:x.size(dim=1), :]
        return self.dropout(res)

