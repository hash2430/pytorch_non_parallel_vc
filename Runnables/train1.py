import os
from hparams import hparam as hp
from models import Net1
import torch
import torch.nn as nn
from net1_train_eval import model_train as net1_train
from dataset import Net1TimitData
from torch.utils.data import DataLoader
from utils import load_logdir, get_logger
import time
from mixture_loss import MaskedCrossEntropyLoss

# No argument from command line.
# Settings from yaml only
# TODO: Create and utilize logger
'''
Call net1_train_eval.model_train()
'''
def train(logdir_train1):
    # load model
    net1_model = Net1(hp.default.phns_len)
    optimizer = torch.optim.Adam(net1_model.parameters(), lr = hp.train1.lr)
    checkpoint_path = '{}/checkpoint.tar'.format(logdir_train1)
    checkpoint = None
    epoch = 0
    loss = 100.0
    lr = hp.train1.lr
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    if checkpoint:
        net1_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.param_groups[0]['lr']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.debug("Checkpoint loaded")
    net1_model.train() # Set to train mode

    # create train/valid loader
    training_set = Net1TimitData(hp.train1.data_path)
    training_loader = DataLoader(training_set, batch_size=hp.train1.batch_size,
                                 shuffle=True, drop_last=True, num_workers=hp.train1.num_workers)
    logger.debug("Training loader created. Size: {} samples".format(training_set.size))

    validation_set = Net1TimitData(hp.eval1.data_path)
    validation_loader = DataLoader(validation_set, batch_size=hp.eval1.batch_size,
                                   shuffle=True, drop_last=False, num_workers=hp.eval1.num_workers)
    logger.debug("Evaluation loader created. Size: {} samples".format(validation_set.size))
    # create criterion
    criterion = MaskedCrossEntropyLoss()
    logger.debug("Loss type: Masked Cross Entropy Loss")

    # run model
    net1_model, epoch, loss = net1_train(checkpoint_path, net1_model, training_loader, validation_loader,
                                         criterion, epoch, device=hp.train1.device,
                                         lr=lr,
                                         loss=loss)

if __name__ == '__main__':
    config_name = 'config'
    hp.set_hparam_yaml(config_name)
    log_dict = load_logdir()
    logger = get_logger('train1', log_dict['train1'])
    logger.info("Training of Network1 starts")
    logger.info('configuration: {}, logdir: {}'.format(config_name, log_dict['train1']))
    logger.info(hp.hparams_debug_string(hp.train1))
    start = time.time()
    train(log_dict['train1'])
    logger.info("Done")
    end = time.time()
    logger.info("Training time: {} s".format(end-start))