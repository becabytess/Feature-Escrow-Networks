import torch
import torch.nn as nn

class BaselineRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, kind):
        super().__init__()
        self.kind = kind

        if kind == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        elif kind == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(kind)

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_stats=False):
        out, state = self.rnn(x)

        if self.kind == "lstm":
            h = state[0][-1]
        else:
            h = state[-1]

        logits = self.head(h)

        if return_stats:
            return logits, {}
        return logits
