import torch
import torch.nn as nn

class RNNPolicy(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=6):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()  # Ensure positive outputs

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return self.activation(out), hidden
