from utils import load_logdir
from models import Net1, Net2
from hparams import hparam as hp
import torch
from net2_train_eval_convert import convert
from dataset import Net2Data
from torch.utils.data import DataLoader

def convert(logdir_eval1, logdir_eval2):
    # Load model net1
    net1_model = Net1()
    checkpoint_path1 = '{}/checkpoint.tar'.format(logdir_eval1)
    checkpoint1 = torch.load(checkpoint_path1)
    if checkpoint1:
        net1_model.load_state_dict(checkpoint1['model_state_dict'])

    # Load model net2
    net2_model = Net2()
    checkpoint_path2 = '{}/checkpoint.tar'.format(logdir_eval2)
    checkpoint2 = torch.load(checkpoint_path2)
    if checkpoint2:
        net2_model.load_state_dict(checkpoint2['model_state_dict'])

    # Create conversion source loader
    conversion_source_set = Net2Data(hp.convert.data_path)
    conversion_source_loader = DataLoader(conversion_source_set,
                                          batch_size=hp.convert.batch_size, shuffle=False, drop_last=False)

    # Run model
    spectrogram_batch = convert(net1_model, net2_model, conversion_source_loader)

    # Postprocessing

    # Save audio


if __name__ == '__main__':
    config_name = 'exp_190121'
    hp.set_hparam_yaml(config_name)
    logdir_dict = load_logdir()

    print('exp: {}, logdir: {}'.format(config_name, logdir_dict['eval2']))
    convert(logdir_dict['eval1'], logdir_dict['eval2'])
    print("Done")