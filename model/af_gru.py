import torch
import torch.nn as nn
from torch.autograd import Variable


class AirdataForecastGRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_len, bidirectional=True):
        super(AirdataForecastGRU, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        h_0 = Variable(torch.zeros(num_layers, x.size(0), self.hidden_size))

        dim = (x.shape[0], self.seq_len, self.input_size)
        x = torch.reshape(x, dim)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out
