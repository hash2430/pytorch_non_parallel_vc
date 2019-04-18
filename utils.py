# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
import logging
import logging.config
import os
import time
# import tfplot
from datetime import datetime
from random import shuffle

import numpy as np
import torch
import yaml

from hparams import hparam as hp


def split_path(path):
    '''
    'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: filepath = 'a/b/c.wav'
    :return: basename, filename, and extension = ('a/b', 'c', 'wav')
    '''
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


def remove_all_files(path):
    files = glob.glob('{}/*'.format(path))
    for f in files:
        os.remove(f)


def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    return normalized


def denormalize_0_1(normalized, max, min):
    values =  np.clip(normalized, 0, 1) * (max - min) + min
    return values


# def plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='confusion_matrix', normalize=False):
#     '''
#     Parameters:
#         correct_labels                  : These are your true classification categories.
#         predict_labels                  : These are you predicted classification categories
#         labels                          : This is a list of labels which will be used to display the axix labels
#         title='Confusion matrix'        : Title for your matrix
#         tensor_name = 'MyFigure/image'  : Name for the output summay tensor
#
#     Returns:
#         summary: TensorFlow summary
#
#     Other itema to note:
#         - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
#         - Currently, some of the ticks dont line up due to rotations.
#     '''
#     cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
#     if normalize:
#         cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
#         cm = np.nan_to_num(cm, copy=True)
#         cm = cm.astype('int')
#
#     np.set_printoptions(precision=2)
#     ###fig, ax = matplotlib.figure.Figure()
#
#     fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
#     ax = fig.add_subplot(1, 1, 1)
#     im = ax.imshow(cm, cmap='Oranges')
#
#     classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
#     classes = ['\n'.join(wrap(l, 40)) for l in classes]
#
#     tick_marks = np.arange(len(classes))
#
#     ax.set_xlabel('Predicted', fontsize=7)
#     ax.set_xticks(tick_marks)
#     c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
#     ax.xaxis.set_label_position('bottom')
#     ax.xaxis.tick_bottom()
#
#     ax.set_ylabel('True Label', fontsize=7)
#     ax.set_yticks(tick_marks)
#     ax.set_yticklabels(classes, fontsize=4, va='center')
#     ax.yaxis.set_label_position('left')
#     ax.yaxis.tick_left()
#
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
#                 verticalalignment='center', color="black")
#     fig.set_tight_layout(True)
#     summary = tfplot.figure.to_summary(fig, tag=tensor_name)
#     return summary

def load_logdir():
    train1_name = hp.train1.exp_name
    train2_name = hp.train2.exp_name
    train3_name = hp.train3.exp_name
    eval1_name = hp.eval1.exp_name
    eval2_name = hp.eval2.exp_name
    eval3_name = hp.eval3.exp_name
    convert_name = 'convert_{}'.format(datetime.now())
    quick_convert_name = 'quick_convert_{}'.format(datetime.now())

    train1_logdir = './logdir/{}/train1'.format(train1_name)
    train2_logdir = './logdir/{}/train2'.format(train2_name)
    train3_logdir = './logdir/{}/train3'.format(train3_name)
    eval1_logdir = './logdir/{}/train1'.format(eval1_name)
    eval2_logdir = './logdir/{}/train2'.format(eval2_name)
    eval3_logdir = './logdir/{}/train3'.format(eval3_name)
    convert_logdir = os.path.join(train2_logdir,convert_name)
    quick_convert_logdir = os.path.join(train3_logdir, quick_convert_name)

    dict = {}
    dict['train1'] = train1_logdir
    dict['train2'] = train2_logdir
    dict['train3'] = train3_logdir
    dict['eval1'] = eval1_logdir
    dict['eval2'] = eval2_logdir
    dict['eval3'] = eval3_logdir
    dict['convert'] = convert_logdir
    dict['quick_convert'] = quick_convert_logdir

    return dict

def get_logger( logger_name, logdir):
    os.makedirs(logdir, exist_ok=True)

    with open('./configs/logging.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date = datetime.now()
    dt = date.date()
    file_handler = logging.handlers.RotatingFileHandler('{}/log_{}.log'.format(logdir, dt), maxBytes=10*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def generate_data_list(logdir, data_path, pctg1, pctg2, pctg3):
    # randomly select and divide into 3 groups of given size
    total_files = glob.glob(data_path)
    shuffle(total_files)
    total_size = len(total_files)
    size1 = int(total_size * pctg1)
    size2 = int(total_size * pctg2)
    size3 = int(total_size * pctg3)

    training_set = total_files[0:size1]
    evaluation_set = total_files[size1:size1+size2]
    test_set = total_files[size1+size2:size1+size2+size3]
    # save the list file to the logdir
    train_list_path = '{}/train_list.txt'.format(logdir)
    eval_list_path = '{}/eval_list.txt'.format(logdir)
    test_list_path = '{}/test_list.txt'.format(logdir)

    with open(train_list_path, 'w') as f1:
        for line in training_set:
            f1.write('{}\n'.format(line))
    with open(eval_list_path, 'w') as f2:
        for line in evaluation_set:
            f2.write('{}\n'.format(line))
    with open(test_list_path, 'w') as f3:
        for line in test_set:
            f3.write('{}\n'.format(line))

    # return the train/eval/test lists
    return training_set, evaluation_set, test_set

def load_train_eval_lists(logdir):
    train_list_path = '{}/train_list.txt'.format(logdir)
    eval_list_path = '{}/eval_list.txt'.format(logdir)

    train_list = []
    eval_list = []

    with open(train_list_path, 'r') as f1:
        for line in f1:
            train_list.append(line.rstrip())

    with open(eval_list_path, 'r') as f2:
        for line in f2:
         eval_list.append(line.rstrip())

    return train_list, eval_list

class Profiler():
    def __init__(self, name):
        self.__name = name
        self.__total = 0
        self.__counting = False

    def start(self):
        if self.__counting:
            raise RuntimeError("Profiler error!!! {} has already been started.".format(self.__name))
        self.__start = time.time()
        self.__counting = True

    def end(self):
        if not self.__counting:
            raise RuntimeError("Profiler error!!! {} must start before calling Profiler.end()".format(self.__name))
        self.__end = time.time()
        diff = self.__end - self.__start
        self.__total += diff
        self.__counting = False

    def get_time(self):
        return self.__total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)