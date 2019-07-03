import time
import os
import torch
from torch.utils.data import DataLoader

from dataset import Net3DataDir
from hparams import hparam as hp
from models import Net3
from net3_train_eval_convert import quick_convert
from utils import load_logdir, get_logger, change_speaker,count_parameters


def convert(logdir_train3, logdir_convert):
    # WARNING! Do not load net1 or net2
    # Load model net3 only
    net3_model = Net3()
    checkpoint_path = '{}/checkpoint.tar'.format(logdir_train3)
    checkpoint = torch.load(checkpoint_path)
    if checkpoint:
        net3_model.load_state_dict(checkpoint['model_state_dict'])

    # Create valid loader
    conversion_source_path = os.path.join(hp.quick_convert.data_path, 'test')
    conversion_source_set = Net3DataDir(conversion_source_path)
    conversion_source_loader = DataLoader(conversion_source_set,
                                          batch_size=hp.quick_convert.batch_size,
                                          shuffle=False,
                                          drop_last=False)

    # Run model: Warning! only give the net3_model, not othermodels here.
    spectrogram_batch = quick_convert(net3_model, conversion_source_loader, logdir_convert)

    # logging
    net3_num_params = count_parameters(net3_model)
    logger.debug('Network 3 number of params: {}'.format(net3_num_params))

if __name__ == '__main__':
    config_name = 'config'
    hp.set_hparam_yaml(config_name)
    logdir_dict = load_logdir()
    logger = get_logger('QUICK_CONVERT', logdir_dict['quick_convert'])
    logger.info('configuration: {}, logdir: {}'.format(config_name, logdir_dict['train3']))
    logger.info(hp.hparams_debug_string(hp.default))
    logger.info(hp.hparams_debug_string(hp.train1))
    logger.info(hp.hparams_debug_string(hp.train2))
    logger.info(hp.hparams_debug_string(hp.train3))
    logger.info(hp.hparams_debug_string(hp.quick_convert))
    start = time.time()
    convert(logdir_dict['train3'], logdir_dict['quick_convert'])
    end = time.time()
    logger.info("Done")
    logger.info("Quick conversion took {} s".format(end - start))