# Models.py: for Net1, Net2, Net3
# modules.py: for CBHG, Conv1d, Highway network, GRU, etc.
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import hparam as hp
class conv1d_instance_norm(nn.Module):
    def __init__(self, in_channels, num_units, kernel_size, stride_size, padding):
        super(conv1d_instance_norm, self).__init__()
        out_channels = num_units
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride_size,
                      padding=padding),
            # num_features:  C from an expected input of size (N, C, L)
            nn.InstanceNorm1d(num_features=out_channels, eps=1e-8, momentum=0.001)
        )

    def forward(self, x):
        output = self.network(x)
        return output


# in_channels: feature dimension
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
            output = nn.functional.relu(self.moduleList[i](x))
            if i == 0:
                l = output.data.shape[-1]
            output = output[:,:,:l]
            outputs.append(output)

        # TODO: concatenate at the last dimension to give (N, Hp.embed_size*K//32, T)
        output_shape = output.size() # (N, C, T)
        outputs = torch.stack(outputs) # (K, N, C, T)
        outputs = outputs.view(output_shape[0], output_shape[1] * len(outputs), output_shape[2])
        return outputs

class prenet(nn.Module):
    def __init__(self, in_dim, hidden_units1, hidden_units2, dropout_rate):
        super(prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(in_dim, hidden_units1) # (N,T,in_features) => (N, T, out_features)
        self.fc2 = nn.Linear(hidden_units1, hidden_units2)
    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.dropout(output, p=self.dropout_rate)
        output = F.relu(self.fc2(output))
        output = F.dropout(output, p=self.dropout_rate)

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

    def forward(self, x):
        h = self.H(x)
        t = self.T(x)
        t_ = 1. - t
        output = h * t + x * t_
        return output

class CBHG(nn.Module):
    def __init__(self, in_channels, num_banks, hidden_units, num_highway_blocks,
                 kernel_size, stride_size, padding_size):
        super(CBHG, self).__init__()
        self.num_highway_blocks = num_highway_blocks
        self.conv1d_banks = conv1d_banks(in_channels=in_channels,
                              num_units = hidden_units,K=num_banks)
        self.maxPool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.conv1d = conv1d_instance_norm(in_channels * num_banks, hidden_units,
                                           kernel_size=kernel_size, stride_size=stride_size, padding=padding_size)
        self.conv1d2 = nn.Conv1d(in_channels, hidden_units,
                                 kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.highwaynet = highwaynet(in_channels, hidden_units)
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_units,bidirectional=True)

    def forward(self, input):
        output = self.conv1d_banks(input) # (N, K * E / 2, T)
        output = self.maxPool1d(output) # (N, K * E / 2, t) # Maxpool on T dim
        output = F.relu(self.conv1d(output)) # (N, E / 2, T)
        output = self.conv1d2(output) # (N, E/2, T)
        output += input # residual connections # (N, E/2, T)

        # Reshape: Highway intput must be in shape (N, *, in_features)
        output = output.transpose(2,1)
        # Highway
        for i in range(self.num_highway_blocks):
            output = self.highwaynet(output) # (N, T, E/2)

        # Reshape: GRU input comes in the shape of (seq_len, batch, input_size) (T, N, E/2)
        output = output.transpose(1, 0)
        # GRU
        output, _ = self.gru(output) # (T, N, E)

        return output
