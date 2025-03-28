import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, unpad_sequence

class SignToGlossModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, device, layers=3):
        super(SignToGlossModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        
        self.lstm = nn.LSTM(input_size, d_model, layers, batch_first=True, bidirectional=False).to(device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        ).to(device)
        
        
    def forward(self, x, seq_len, device, h=None, c=None):
        # h = torch.zeros(, x.shape[0], self.d_model).to(device) if h is None else h
        # c = torch.zeros(self.lstm.num_layers, x.shape[0], self.d_model).to(device) if c is None else c
        
        packed = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False).to(device)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))

        return self.ff(out)