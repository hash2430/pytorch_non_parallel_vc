import torch.nn as nn
import modules
import torch.nn.functional as F
from hparams import hparam as hp
import torch

class Net1(nn.Module):
    def __init__(self, phns_len):
        super(Net1, self).__init__()
        self.prenet = modules.prenet(hp.default.n_mfcc,
                                     hp.train1.hidden_units,
                                     hp.train1.hidden_units // 2,
                                     hp.train1.dropout_rate)
        self.CBHG = modules.CBHG(hp.train1.hidden_units // 2,
                                 hp.train1.num_banks,
                                 hp.train1.hidden_units // 2,
                                 hp.train1.num_highway_blocks,
                                 hp.train1.kernel_size,
                                 hp.train1.stride_size,
                                 hp.train1.padding_size)
        self.fc = nn.Linear(hp.train1.hidden_units, phns_len)
    def forward(self, mfccs):
        prenet_out = self.prenet(mfccs) # (N, E/2, T)
        prenet_out = prenet_out.transpose(2,1) # (N, T, E/2)
        out = self.CBHG(prenet_out) # (T, N, E)
        # Reshape: FC needs (N, T, E) as input
        out = out.transpose(1, 0)
        logits = self.fc(out) # (N, T, P)
        ppgs = F.softmax(logits, dim=-1) # (N, T, P)
        # preds: arg max index of ppgs (1, 61)
        preds = ppgs.data.max(2)[1] # (N, T)
        return ppgs, preds, logits

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.prenet = modules.prenet(hp.default.phns_len,
                                     hp.train2.hidden_units,
                                     hp.train2.hidden_units // 2,
                                     hp.train2.dropout_rate
                                     )
        self.cbhg1 = modules.CBHG(hp.train2.hidden_units // 2,
                                  hp.train2.num_banks,
                                  hp.train2.hidden_units // 2,
                                  hp.train2.num_highway_blocks,
                                  hp.train2.kernel_size,
                                  hp.train2.stride_size,
                                  hp.train2.padding_size
                                  )
        self.densenet1 = nn.Linear(hp.train2.hidden_units, hp.default.n_mels)

        # Originally, this layer was meant to FC to n_mel dimension for dim compression effect
        self.densenet2 = nn.Linear(hp.default.n_mels,
                                   hp.train2.hidden_units // 2
                                   )
        self.cbhg2 = modules.CBHG(hp.train2.hidden_units // 2,
                                  hp.train2.num_banks,
                                  hp.train2.hidden_units // 2,
                                  hp.train2.num_highway_blocks,
                                  hp.train2.kernel_size,
                                  hp.train2.stride_size,
                                  hp.train2.padding_size
                                  )
        self.n_bins = 1 + hp.default.n_fft//2 # 257
        self.n_units = self.n_bins * hp.train2.n_mixtures #(1285) predict pdf parameter for each frequency
        # TODO: implement pytorch equivalebt for bias_initializer
        self.densenet3 = nn.Linear(hp.train2.hidden_units,
                                   hp.default.n_fft//2+1) # (n, T, 257)

    def forward(self, ppgs): # ppgs (N, T, V)
        out = self.prenet(ppgs) # (N, T, E/2)

        out = out.transpose(2, 1) # (N, C, T)
        out = self.cbhg1(out) # (T, N, E)
        out = out.transpose(1, 0) # (N, T, E)
        pred_mel = self.densenet1(out) # (N, T, n_mel)

        out = self.densenet2(pred_mel) # (N, T, E/2)
        # Reshape: cbhg input needs shape
        out = out.transpose(2, 1) # (N, E/2, T)
        out = self.cbhg2(out) # (T, N, E)
        # Reshape: FC input needs shape
        out = out.transpose(1, 0) # (N, T, E)
        pred_spec = self.densenet3(out) # (N, T, 257)
        return pred_spec, pred_mel # Both in log scale

# Is it the right approach to have net1 and net2 as submodule of Net3?
class Net3(nn.Module):
    # Define Net1 and Net2 as submodule
    def __init__(self):
        super(Net3, self).__init__()
        self.prenet = modules.prenet(hp.default.n_mfcc,
                                     hp.train3.hidden_units,
                                     hp.train3.hidden_units // 2,
                                     hp.train3.dropout_rate
                                    )

        self.cbhg1 = modules.CBHG(hp.train3.hidden_units // 2,
                                  hp.train3.num_banks,
                                  hp.train3.hidden_units // 2,
                                  hp.train3.num_highway_blocks,
                                  hp.train3.kernel_size,
                                  hp.train3.stride_size,
                                  hp.train3.padding_size
                                  )
        self.densenet1 = nn.Linear(hp.train2.hidden_units,
                                   hp.default.n_mels)

        self.densenet2 = nn.Linear(hp.default.n_mels, hp.train2.hidden_units // 2)

        self.cbhg2 = modules.CBHG(hp.train3.hidden_units // 2,
                                  hp.train3.num_banks,
                                  hp.train3.hidden_units // 2,
                                  hp.train3.num_highway_blocks,
                                  hp.train3.kernel_size,
                                  hp.train3.stride_size,
                                  hp.train3.padding_size
                                  )

        self.n_bins = 1 + hp.default.n_fft // 2
        self.n_units = self.n_bins * hp.train3.n_mixtures
        self.densenet3 = nn.Linear(hp.train3.hidden_units,
                                   hp.default.n_fft//2 + 1)
    def forward(self, mfccs):
        out = self.prenet(mfccs) # (N, T, E/2)

        out = out.transpose(2, 1)
        out = self.cbhg1(out)
        out = out.transpose(1, 0)
        pred_mel = self.densenet1(out) # (N, T, E/2)

        out = self.densenet2(pred_mel)
        # Reshape: cbhg input needs shape (N, E/2, T)
        out = out.transpose(2, 1)
        out = self.cbhg2(out) # (T, N, E)
        # Reshape: FC input needs shape (N, T, E)
        out = out.transpose(1, 0)
        pred_spec = self.densenet3(out) # (N, T, n_bins x n_mixtures x 3)
        return pred_spec, pred_mel