from hparams import hparam as hp
from models import Net1, Net2, Net3
import torch
import torch.nn as nn
import os
from net3_train_eval_convert import model_train as net3_train
from dataset import Net3Data
from torch.utils.data import DataLoader
from utils import load_logdir, get_logger, load_train_eval_lists, generate_data_list
import time
from mixture_loss import MyMSELoss

'''
Call net3_train_eval_convert.model_train()
'''
def train(logdir_train1, logdir_train2, logdir_train3):
    # Load model Net1 for evaluation
    net1_model = Net1(hp.default.phns_len)
    checkpoint_path1 = '{}/checkpoint.tar'.format(logdir_train1)
    checkpoint1 = torch.load(checkpoint_path1)
    if checkpoint1:
        net1_model.load_state_dict(checkpoint1['model_state_dict'])

    # Load model Net2 for evaluation
    net2_model = Net2()
    checkpoint_path2 = '{}/checkpoint.tar'.format(logdir_train2)
    checkpoint2 = torch.load(checkpoint_path2)
    if checkpoint2:
        net2_model.load_state_dict(checkpoint2['model_state_dict'])

    # Load model Net3 for training
    net3_model = Net3()
    optimizer = torch.optim.Adam(net3_model.parameters(), lr=hp.train3.lr)
    checkpoint_path3 = '{}/checkpoint.tar'.format(logdir_train3)
    checkpoint3 = None
    if os.path.exists(checkpoint_path3):
        checkpoint3 = torch.load(checkpoint_path3)
    epoch = 0
    loss = 100.0
    lr = hp.train3.lr
    if checkpoint3:
        train_list, eval_list = load_train_eval_lists(logdir_train3)
        logger.info("Reuse existing train_list, eval_list from {}".format(logdir_train3))
        net3_model.load_state_dict(checkpoint3['model_state_dict'])
        optimizer.load_state_dict(checkpoint3['optimizer_state_dict'])
        lr = optimizer.param_groups[0]['lr']
        epoch = checkpoint3['epoch']
        loss = checkpoint3['loss']
    else:
        data_dir = hp.train3.data_path
        train_list, eval_list, _ = generate_data_list(logdir_train3, data_dir, 0.8, 0.1, 0.1)
        logger.info("Generate new train_list, eval_list, test_list.")
    net3_model.train() # Set to train mode

    # Create train/valid loader
    training_set = Net3Data(train_list)
    training_loader = DataLoader(training_set, batch_size=hp.train3.batch_size,
                                 shuffle=True, drop_last=True, num_workers=hp.train3.num_workers)
    logger.debug("Training loader created. Size: {} samples".format(training_set.size))
    validation_set = Net3Data(eval_list)
    validation_loader = DataLoader(validation_set, batch_size=hp.train3.batch_size,
                                   shuffle=True, drop_last=True, num_workers=hp.eval3.num_workers)
    logger.debug("Validation loader created. Size: {}".format(validation_set.size))
    # Create criterion
    criterion = MyMSELoss()
    logger.debug("Loss type: MSE loss on linear and mel-spectrogram")
    # Run model
    net3_model,_,_ = net3_train(checkpoint_path3, net1_model, net2_model, net3_model,
                            training_loader, validation_loader, criterion,starting_epoch=epoch,
                            device=hp.train3.device,
                            lr=lr,
                            loss=loss)

if __name__ == '__main__':
    config_name = 'config_190213'
    hp.set_hparam_yaml(config_name)
    log_dict = load_logdir()
    logger = get_logger('train3', log_dict['train3'])
    logger.info("Training of Network3 starts")
    logger.info('configuration: {}, logdir: {}'.format(config_name, log_dict['train3']))
    start = time.time()
    train(log_dict['train1'], log_dict['train2'], log_dict['train3'])
    end = time.time()
    logger.info("Done")
    logger.info("Training time: {} s".format(end - start))