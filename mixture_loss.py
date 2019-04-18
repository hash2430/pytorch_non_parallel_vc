from hparams import hparam as hp
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_utils import pdf_parameter_extraction

# TODO: torch.where implementation does not exist for v0.3.0 So the log_prob normalization for edge case is excluded for now
class MixtureLoss(nn.Module):
    def __init__(self, n_mixtures, batch_size, mol_step, n_fft):
        super(MixtureLoss, self).__init__()
        self.n_mixtures = n_mixtures
        self.batch_size = batch_size
        self.mol_step = mol_step
        self.n_fft = n_fft
    def forward(self, x, y):
        mu, log_var, log_pi = pdf_parameter_extraction(x, self.n_mixtures, self.batch_size)
        y = y.repeat(1, 1, self.n_mixtures)
        y = y.view(self.batch_size, -1, self.n_fft//2+1, self.n_mixtures)
        centered_x = y - mu
        inv_stdv = torch.exp(-log_var)
        plus_in = inv_stdv * (centered_x + self.mol_step)
        min_in = inv_stdv * (centered_x - self.mol_step)
        cdf_plus = F.sigmoid(plus_in)
        cdf_min = F.sigmoid(min_in)

        # log probability for edge case
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min
        clamped_cdf_delta = torch.clamp(cdf_delta, min=1e-12)
        clamped_cdf_delta = torch.log(clamped_cdf_delta)
        # temp = clamped_cdf_delta
        # if y > 0.999:
        #     temp = log_one_minus_cdf_min
        #
        # log_prob = temp
        # if y < 0.001:
        #     log_prob = log_cdf_plus

        # temp = torch.where(y > 0.999, log_one_minus_cdf_min, clamped_cdf_delta)
        # log_prob = torch.where(y < 0.001, log_cdf_plus, temp)
        log_prob = clamped_cdf_delta
        log_prob = log_prob + log_pi
        prob = torch.exp(log_prob)
        prob = torch.sum(prob, dim=-1)
        log_prob = torch.log(prob)
        loss_mle = -torch.mean(log_prob)
        return loss_mle

class MyMSELoss(nn.Module):
    def __init__(self):
        super(MyMSELoss, self).__init__()

    def forward(self, pred_spec, pred_mel, y_spec, y_mel):
        spec_loss = nn.MSELoss()
        spec_loss = spec_loss(pred_spec, y_spec)
        mel_loss = nn.MSELoss()
        mel_loss = mel_loss(pred_mel, y_mel)
        loss = spec_loss + mel_loss
        return loss

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, mfcc, preds, y_ppgs):
        dtype = torch.FloatTensor
        if hp.train1.device != 0:
            dtype = torch.cuda.FloatTensor
        is_target = torch.sign(torch.abs(torch.sum(mfcc, dim=-1))).view(-1)
        ce = torch.nn.CrossEntropyLoss(reduce=False)
        loss = ce(preds, y_ppgs)
        loss = torch.mul(is_target, loss)
        loss = torch.mean(loss)
        return loss

