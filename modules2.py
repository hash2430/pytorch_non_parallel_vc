import torch
import torch.nn as nn
import torch.nn.functional as F

class conv1d_instance_norm(nn.Module):
    def __init__(self, in_channels, num_units, kernel_size, stride_size, padding):
        super(conv1d_instance_norm, self).__init__()
        out_channels = num_units
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride_size,
                      padding=padding),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, input):
        output = self.network(input)
        return output

class conv1d_banks(nn.Module):
    def __init__(self, in_channels, num_units, K=8):
        super(conv1d_banks, self).__init__()
        self.moduleList = nn.ModuleList()
        for i in range(1, K+1):
            m = conv1d_instance_norm(in_channels, num_units, i, 1, i//2)
            self.moduleList.append(m)
    def forward(self, x):
        outputs = []
        # Call convolution banks
        # Same input x to all the convolutions (parallel relationship, not sequential between convolution banks)
        for i, m in enumerate(self.moduleList):
            output = m(x)
            output = F.relu(output)
            if i == 0:
                l = output.data.shape[-1]
            output = output[:,:,:l]
            outputs.append(output)

        # TODO: concatenate at the last dimension to give (N, Hp.embed_size*K//32, T)
        output_shape = output.size() # (N, C, T)
        outputs = torch.stack(outputs) # (K, N, C, T)
        num_banks = len(outputs)
        outputs = outputs.transpose(1, 0) # (N, K, C, T)
        outputs = outputs.contiguous().view(output_shape[0], output_shape[1] * num_banks, output_shape[2])
        return outputs
class prenet(nn.Module):
    def __init__(self, in_dim, hidden_units1, hidden_units2, dropout_rate):
        super(prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(in_dim, hidden_units1)
        self.fc2 = nn.Linear(hidden_units1, hidden_units2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_units1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_units2)
    def forward(self, x):
        output = self.fc1(x)
        output = F.dropout(output, p=self.dropout_rate).transpose(1, 2)
        output = self.batch_norm1(output)
        output = F.relu(output).transpose(1, 2)

        output = self.fc2(output)
        output = F.dropout(output, p=self.dropout_rate).transpose(1, 2)
        output = self.batch_norm2(output)
        output = F.relu(output).transpose(1, 2)

        return output

class highwaynet(nn.Module):
    def __init__(self, in_features, num_units):
        super(highwaynet, self).__init__()
        self.H = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=num_units),
            nn.ReLU()
        )
        self.T = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=num_units, bias=True),
            nn.Sigmoid()
        )
        self.batch_norm = nn.BatchNorm1d(num_units)
    def forward(self, x):
        h = self.H(x)
        t = self.T(x)
        t_ = 1. - t
        output = h * t + x * t_
        output = self.batch_norm(output.transpose(1, 2)).transpose(1, 2)
        return output

class CBHG(nn.Module):
    def __init__(self, in_channels, num_banks, hidden_units,
               num_highway_blocks, kernel_size, stride_size, padding_size):
        super(CBHG, self).__init__()
        self.num_highway_blocks = num_highway_blocks
        self.conv1d_banks = conv1d_banks(in_channels=in_channels,
                                         num_units=hidden_units, K=num_banks)
        self.maxPool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.conv1d = nn.Sequential(nn.Conv1d(in_channels * num_banks, hidden_units,
                                           kernel_size=kernel_size, stride=stride_size,
                                              padding=padding_size),
                                    nn.BatchNorm1d(hidden_units),
                                    nn.ReLU())
        # conv1d2 removed
        self.highwaynet = highwaynet(hidden_units, hidden_units)
        self.gru = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, bidirectional=True)

    def forward(self, input):
        output = self.conv1d_banks(input) #(N, K*E/2, T)
        output = self.maxPool1d(output)
        output = self.conv1d(output)
        output += input # residual connections

        # Reshape: Highway input must be in shape (N, T, C)
        output = output.transpose(2, 1)
        # Highway
        for i in range(self.num_highway_blocks):
            output = self.highwaynet(output) # (N, T, E/2)

        # Reshpae: GRU intput comes in the shape of (T, N, C)
        output = output.transpose(1, 0)

        # GRU
        output, _ = self.gru(output)
        return output