# stdlib
import time
import itertools
import sys
import math
import h5py

# external
from scipy import misc
import torch
import numpy as np
from torch.autograd import Variable

# local
from options.train_options import TrainOptions
from models.hourglass_model import HourglassModel
from data.data_loader import CreateDataLoader

# broken local imports
# from data.data_loader import CreateDIWDataLoader
# import models.networks

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = "/"

train_list_dir_landscape = root + '/phoenix/S6/zl548/MegaDpeth_code/train_list/landscape/'
input_height = 240
input_width = 320
data_loader_l = CreateDataLoader(root, train_list_dir_landscape, input_height, input_width)
dataset_l = data_loader_l.load_data()
dataset_size_l = len(data_loader_l)
print('========================= training landscape  images = %d' % dataset_size_l)

train_list_dir_portrait = root + '/phoenix/S6/zl548/MegaDpeth_code/train_list/portrait/'
input_height = 320
input_width = 240
data_loader_p = CreateDataLoader(root, train_list_dir_portrait, input_height, input_width)
dataset_p = data_loader_p.load_data()
dataset_size_p = len(data_loader_p)
print('========================= training portrait  images = %d' % dataset_size_p)

_isTrain = False
batch_size = 32
num_iterations_L = dataset_size_l / batch_size
num_iterations_P = dataset_size_p / batch_size
model = HourglassModel(opt)  # , _isTrain  # function only takes 1 param
model.switch_to_train()

best_loss = 100

print("num_iterations ", num_iterations_L, num_iterations_P)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
total_iteration = 0

print("=================================  BEGIN VALIDATION =====================================")

# best_loss = validation(model, test_dataset, test_dataset_size)  # todo investigate
# print("best_loss  ", best_loss)

start_diw_idx = -1
valiation_interval = 300

for epoch in range(0, 20):

    if epoch > 0:
        model.update_learning_rate()

    # landscape 
    for i, data in enumerate(dataset_l):
        total_iteration = total_iteration + 1
        stacked_img = data['img_1']
        targets = data['target_1']
        is_DIW = False
        # model.set_input(stacked_img, targets, is_DIW)  # warning: unexpected argument
        # model.optimize_parameters(epoch)

    # portrait 
    for i, data in enumerate(dataset_p):
        total_iteration = total_iteration + 1
        stacked_img = data['img_1']
        targets = data['target_1']
        is_DIW = False
        # model.set_input(stacked_img, targets, is_DIW)  # warning: unexpected argument
        # model.optimize_parameters(epoch)

print("We are done")
