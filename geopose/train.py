# stdlib
from collections.abc import Iterable
from unittest.mock import patch
import csv
import gzip
import h5py
import itertools
import math
import os
import pathlib
import shutil
import sys
import time

# external
from scipy import misc
from skimage import io
from skimage.transform import resize
from skimage.transform import resize
from torch.autograd import Variable
from torchvision import transforms
import cv2 as cv
import imageio  # read PFM
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time
import torch
import torchvision.transforms

# local
from megadepth.data.data_loader import CreateDataLoader
from megadepth.models.HG_model import HGModel
from megadepth.models.models import create_model
from megadepth.options.eval_options import EvalOptions
from megadepth.options.train_options import TrainOptions
from semseg.models.models import ModelBuilder, SegmentationModule
from semseg.utils import colorEncode
from utils.process_images import get_sky_mask, transform_image_for_megadepth, megadepth_predict, \
    transform_image_for_semseg, semseg_predict, apply_sky_mask
from utils.semseg import visualize_result
from geopose.dataset import GeoPoseDataset, clear_dataset_dir, rotate_images


class JointLoss(torch.nn.Module):
    """
    todo: align dict keys with dataset (and predictions?)


    https://github.com/ArnaudFickinger/MegaDepth-Training/blob/f3306861670c1fd54335d707e1a47a1e6e95cf11/models/networks.py#L117
    """

    def __init__(self):
        super(JointLoss, self).__init__()
        self.w_data = 1.0
        self.w_grad = 0.5
        self.w_sm = 2.0
        self.w_od = 0.5
        self.w_od_auto = 0.2
        self.w_sky = 0.1
        # self.h_offset = [0,0,0,1,1,2,2,2,1]
        # self.w_offset = [0,1,2,0,2,0,1,2,1]
        self.total_loss = None

    def Ordinal_Loss(self, prediction_d, targets, input_images):
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        n_point_total = 0

        batch_input = torch.exp(prediction_d)

        ground_truth_arr = Variable(targets['ordinal'].cuda(), requires_grad=False)
        # sys.exit()
        for i in range(0, prediction_d.size(0)):
            gt_d = ground_truth_arr[i]
            n_point_total = n_point_total + gt_d.size(0)
            # zero index!!!!
            x_A_arr = targets['x_A'][i]
            y_A_arr = targets['y_A'][i]
            x_B_arr = targets['x_B'][i]
            y_B_arr = targets['y_B'][i]

            inputs = batch_input[i, :, :]

            # o_img = input_images[i,:,:,:].data.cpu().numpy()
            # o_img = np.transpose(o_img, (1,2,0))

            # store_img = inputs.data.cpu().numpy()
            # misc.imsave(targets['path'][i].split('/')[-1] + '_p.jpg', store_img)
            # misc.imsave(targets['path'][i].split('/')[-1] + '_o.jpg', o_img)

            z_A_arr = inputs[y_A_arr, x_A_arr]
            z_B_arr = inputs[y_B_arr, x_B_arr]

            inner_loss = torch.mul(-gt_d, (z_A_arr - z_B_arr))

            if inner_loss.data[0] > 5:
                print('DIW difference is too large !!!!')
                # inner_loss = torch.mul(-gt_d, (torch.log(z_A_arr)   - torch.log(z_B_arr) ) )
                return 5

            ordinal_loss = torch.log(1 + torch.exp(inner_loss))

            total_loss = total_loss + torch.sum(ordinal_loss)

        if total_loss.data[0] != total_loss.data[0]:
            print("SOMETHING WRONG !!!!!!!!!!", total_loss.data[0])
            sys.exit()

        return total_loss / n_point_total

    def sky_loss(self, prediction_d, targets, i):
        tau = 4
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        gt_d = 1
        # inverse depth
        inputs = torch.exp(prediction_d)
        x_A_arr = targets['sky_x'][i, 0]
        y_A_arr = targets['sky_y'][i, 0]
        x_B_arr = targets['depth_x'][i, 0]
        y_B_arr = targets['depth_y'][i, 0]

        z_A_arr = inputs[y_A_arr, x_A_arr]
        z_B_arr = inputs[y_B_arr, x_B_arr]

        inner_loss = -gt_d * (z_A_arr - z_B_arr)

        if inner_loss.data[0] > tau:
            print("sky prediction reverse")
            inner_loss = -gt_d * (torch.log(z_A_arr) - torch.log(z_B_arr))

        ordinal_loss = torch.log(1 + torch.exp(inner_loss))
        return torch.sum(ordinal_loss)

    def Ordinal_Loss_AUTO(self, prediction_d, targets, i):
        tau = 1.2
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        # n_point_total = 0

        inputs = torch.exp(prediction_d)
        gt_d = targets['ordinal'][i, 0]

        x_A_arr = targets['x_A'][i, 0]
        y_A_arr = targets['y_A'][i, 0]
        x_B_arr = targets['x_B'][i, 0]
        y_B_arr = targets['y_B'][i, 0]

        z_A_arr = inputs[y_A_arr, x_A_arr]
        z_B_arr = inputs[y_B_arr, x_B_arr]

        # A is close, B is further away
        inner_loss = -gt_d * (z_A_arr - z_B_arr)

        ratio = torch.div(z_A_arr, z_B_arr)

        if ratio.data[0] > tau:
            print("DIFFERNCE IS TOO LARGE, REMOVE OUTLIERS!!!!!!")
            return 1.3873
        else:
            ordinal_loss = torch.log(1 + torch.exp(inner_loss))
            return torch.sum(ordinal_loss)

    def GradientLoss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)

        v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
        v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
        h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / N

        return gradient_loss

    def Data_Loss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum(torch.pow(log_d_diff, 2)) / N
        s2 = torch.pow(torch.sum(log_d_diff), 2) / (N * N)
        data_loss = s1 - s2

        return data_loss

    def Data_Loss_test(self, prediction_d, targets):
        mask = targets['mask'].cuda()
        d_gt = targets['gt'].cuda()
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        k = 0.5
        for i in range(0, mask.size(0)):
            # number of valid pixels
            N = torch.sum(mask[i, :, :], 0)
            d_log_gt = torch.log(d_gt[i, :, :])
            log_d_diff = prediction_d[i, :, :] - d_log_gt
            log_d_diff = torch.cmul(log_d_diff, mask)

            data_loss = (torch.sum(torch.pow(log_d_diff, 2)) / N - torch.pow(torch.sum(log_d_diff), 2) / (N * N))

            total_loss = total_loss + data_loss

        return total_loss / mask.size(0)

    def __call__(self, input_images, prediction_d, targets, is_DIW, current_epoch):
        # num_features_d = 5

        # prediction_d_un = prediction_d.unsqueeze(1)
        prediction_d_1 = prediction_d[:, ::2, ::2]
        prediction_d_2 = prediction_d_1[:, ::2, ::2]
        prediction_d_3 = prediction_d_2[:, ::2, ::2]

        mask_0 = Variable(targets['mask_0'].cuda(), requires_grad=False)
        d_gt_0 = torch.log(Variable(targets['gt_0'].cuda(), requires_grad=False))

        mask_1 = Variable(targets['mask_1'].cuda(), requires_grad=False)
        d_gt_1 = torch.log(Variable(targets['gt_1'].cuda(), requires_grad=False))

        mask_2 = Variable(targets['mask_2'].cuda(), requires_grad=False)
        d_gt_2 = torch.log(Variable(targets['gt_2'].cuda(), requires_grad=False))

        mask_3 = Variable(targets['mask_3'].cuda(), requires_grad=False)
        d_gt_3 = torch.log(Variable(targets['gt_3'].cuda(), requires_grad=False))

        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        count = 0

        for i in range(0, mask_0.size(0)):
            # print(i, targets['has_ordinal'][i, 0])
            if targets['has_ordinal'][i, 0] > 0.1:
                continue
            else:
                total_loss += self.w_data * self.Data_Loss(prediction_d[i, :, :], mask_0[i, :, :], d_gt_0[i, :, :])
                # this could be 4 scales of gradient loss: -Petr
                total_loss += self.w_grad * self.GradientLoss(prediction_d[i, :, :], mask_0[i, :, :], d_gt_0[i, :, :])
                total_loss += self.w_grad * self.GradientLoss(prediction_d_1[i, :, :], mask_1[i, :, :], d_gt_1[i, :, :])
                total_loss += self.w_grad * self.GradientLoss(prediction_d_2[i, :, :], mask_2[i, :, :], d_gt_2[i, :, :])
                total_loss += self.w_grad * self.GradientLoss(prediction_d_3[i, :, :], mask_3[i, :, :], d_gt_3[i, :, :])
                count += 1

        if count == 0:
            count = 1

        total_loss = total_loss / count

        self.total_loss = total_loss

        return total_loss.data[0]

    def get_loss_var(self):
        return self.total_loss


if __name__ == '__main__':

    megadepth_checkpoints_path = './megadepth/checkpoints/'
    # with patch.object(sys, 'argv', ['/home/petrmiculek/Code/KNN-projekt/geopose/train.py']):
    #     opt = EvalOptions().parse(megadepth_checkpoints_path)

    with patch.object(sys, 'argv', ['/home/petrmiculek/Code/KNN-projekt/geopose/train.py']):
        opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    model = create_model(opt)
    # model = HGModel(opt)

    if False:
        root = "/"

        train_list_dir_landscape = root + '/phoenix/S6/zl548/MegaDpeth_code/train_list/landscape/'
        input_height = 240
        input_width = 320
        data_loader_l = CreateDataLoader(root, train_list_dir_landscape, input_height, input_width)
        dataset_l = data_loader_l.load_data()
        dataset_size_l = len(data_loader_l)
        print('========================= training landscape  images = %d' % dataset_size_l)
    print('...')

    ds_dir = 'datasets/geoPose3K_final_publish/'
    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])
                                         ])

    ds = GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform, verbose=False)

    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate)

    """ Past here nothing has been proven to work """
    # set training mode
    model.netG.train()

    # get loss function

    # get optimizer
    self.optimizer_G = torch.optim.Adam(model.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for i, sample in enumerate(ds[0:5]):
        img, depth, path = sample

        optimizer.zero_grad()
        model.netG.forward(img)
        self.input_images = Variable(self.input.cuda())
        self.prediction_d = self.netG.forward(self.input_images)
