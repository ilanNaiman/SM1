import torch.nn as nn


class ForecastNet(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(ForecastNet, self).__init__()
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, in_features)
        self.loss_f = nn.MSELoss()

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.linear(output)
        return output

    def loss(self, target, output):
        return self.loss_f(output, target)
