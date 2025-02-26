import torch
import torch.nn as nn

class SignToGlossModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, layers=3):
        super(SignToGlossModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        
        self.lstm = nn.LSTM(input_size, d_model, layers, batch_first=True)
        self.ff = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, h=None, c=None):
        h = torch.zero(self.lstm.num_layers, x.shape[0], self.d_model) if h is None else h
        c = torch.zero(self.lstm.num_layers, x.shape[0], self.d_model) if c is None else c

        out, _ = self.lstm(x, (h, c))
        out = out[:, -1]
        return self.softmax(self.ff(out))