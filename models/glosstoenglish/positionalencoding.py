import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_p = 0.1, max_len = 1000, device='cpu'):
        """
        Create a positional encoding matrix.
        
        Parameters:
            max_len: The maximum length a sequence of tokens can be
            d_model: The dimensions of the encoding vector
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_p)

        self.pe = torch.zeros(max_len, d_model).to(device) 
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        dim = torch.arange(0, d_model, 2)
        
        # For even position indices, we utilize the sine function
        # For odd indices, we utilize the cosine function
        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / d_model)))

    def forward(self, x): 
        """
        Adds the positional encoding to the embedding matrix
        Helps describes the position of the embedding vector within a sequence.
        
        Parameters:
            x: Embedding matrix (Batch, Sequence Size, Embedding Size)

        Returns:
            Embedding matrix with positions encoded into them
        """
        
        res = x + self.pe[:x.size(dim=1), :]
        return self.dropout(res)

