import torch
import torch.nn as nn
import torch.nn.functional as F


class myRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_hidden = 256
        self.z_dim = 16
        self.bs = 2

        self.rnn = nn.LSTMCell(input_size=self.z_dim,
                               hidden_size=self.lstm_hidden)
        self.rnn_linear = nn.Sequential(nn.Linear(self.lstm_hidden, 28 * 28),
                                        nn.Sigmoid())

    def forward(self, z):
        # z [step,bs,z_dim]
        c_x = torch.zeros([self.bs, self.lstm_hidden])
        h_x = torch.zeros([self.bs, self.lstm_hidden])

        for step in range(z.shape[0]):
            h_x, c_x = self.rnn(z[step], (h_x, c_x))
            output = self.rnn_linear(h_x)
        return (output)


if __name__ == "__main__":
    RNN = myRNN()
    z = torch.empty([3, 2, 16]).normal_()
    output = RNN(z)
    print(output)
