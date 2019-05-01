from hparams import hparam as hp
from models import Net1, Net2
import torch
print(torch.__version__)
import torch.nn as nn
from net2_train_eval_convert import model_train as net2_train
from dataset import Net2Data
from torch.utils.data import DataLoader
from utils import load_logdir, get_logger, generate_data_list, load_train_eval_lists
import time
import os
from mixture_loss import MyMSELoss

'''
Call net2_train_eval_convert.model_train()
'''
def train(logdir_train1, logdir_train2):
    # Load model net1
    net1_model = Net1(hp.default.phns_len)
    checkpoint_path1 = '{}/checkpoint.tar'.format(logdir_train1)
    checkpoint1 = torch.load(checkpoint_path1)
    if checkpoint1:
        net1_model.load_state_dict(checkpoint1['model_state_dict'])

    # Load model net2
    net2_model = Net2()
    checkpoint_path2 = '{}/checkpoint.tar'.format(logdir_train2)
    checkpoint2 = None
    epoch = 0
    loss = 100
    lr = hp.train2.lr
    optimizer = torch.optim.Adam(net2_model.parameters(), lr=lr)
    if os.path.exists(checkpoint_path2):
        checkpoint2 = torch.load(checkpoint_path2)
    if checkpoint2:
        train_list, eval_list = load_train_eval_lists(logdir_train2)
        logger.info("Reuse existing train_list, eval_list from {}".format(logdir_train2))
        net2_model.load_state_dict(checkpoint2['model_state_dict'])
        optimizer.load_state_dict(checkpoint2['optimizer_state_dict'])
        lr = optimizer.param_groups[0]['lr']
        epoch = checkpoint2['epoch']
        loss = checkpoint2['loss']
        logger.debug("Checkpoint loaded")
    else:
        data_dir = hp.train2.data_path
        train_list, eval_list, _ = generate_data_list(logdir_train2, data_dir, 0.8, 0.1, 0.1)
        logger.info("Generate new train_list, eval_list, test_list.")
    net2_model.train() # Set to train mode

    # Create train/valid loader
    training_set = Net2Data(train_list)
    training_loader = DataLoader(training_set, batch_size=hp.train2.batch_size,
                                 shuffle=False, drop_last=True, num_workers=hp.train2.num_workers)
    logger.debug("Training loader created. Size: {} samples".format(training_set.size))
    validation_set = Net2Data(eval_list)
    # If batch_size is inconsistent at the last batch, audio_utils.net2_out_to_pdf fails
    '''
    TODO: not sure if validation_loader requires separate batch size as 'eval2.batch_size'
    maybe implement later
    '''
    validation_loader = DataLoader(validation_set, batch_size=hp.train2.batch_size,
                                   shuffle=False, drop_last=True, num_workers=hp.eval2.num_workers)
    logger.debug("Validation loader created. Size: {} samples".format(validation_set.size))
    # Create criterion
    criterion = MyMSELoss()
    logger.debug("Loss type: Sum of MSE loss on mel spectrogram and linear spectrogram")
    # Run model
    net2_model, epoch, best_loss = net2_train(checkpoint_path2, net1_model, net2_model,
                            training_loader, validation_loader, criterion, epoch,
                            device=hp.train2.device,
                            lr=lr,
                            loss=loss)

    # Checkpoint saving occurs inside the net2_train
if __name__ == '__main__':
    config_name = 'config'
    hp.set_hparam_yaml(config_name)
    log_dict = load_logdir()
    logger = get_logger('train2', log_dict['train2'])
    logger.info("Training of Network2 starts")
    logger.info('configuration: {}, logdir: {}'.format(config_name, log_dict['train2']))
    logger.info(hp.hparams_debug_string(hp.train2))
    start = time.time()
    train(log_dict['train1'], log_dict['train2'])
    logger.info("Done")
    end = time.time()
    logger.info("Training time: {} s".format(end-start))