import torch.nn as nn


class SignToGlossModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, layers=2):
        super(SignToGlossModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size, hidden_size, layers, batch_first=True, bidirectional=True
        ).to(device)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size),
        ).to(device)

    def forward(self, x, device, h=None, c=None):
        out, _ = self.lstm(x)
        return self.ff(out[:, -1, :])
