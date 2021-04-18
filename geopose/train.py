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


def rmse_loss(pred, gt, mask=None, scale_invariant=True):
    # from rmse_error_main.py
    assert gt.shape == pred.shape, \
        '{} x {}'.format(gt.shape, pred.shape)

    #save_from_nan = 100.0

    #pred = torch.log(pred + save_from_nan)
    #gt = torch.log(gt + save_from_nan)

    if mask is None:
        mask = torch.zeros(pred.shape) + 1

    n = torch.sum(mask)

    diff = pred - gt
    diff = torch.mul(diff, mask)

    s1 = torch.sum(torch.pow(diff, 2)) / n
    s2 = torch.pow(torch.sum(diff), 2) / (n * n)

    if scale_invariant:
        data_loss = s1 - s2
    else:
        data_loss = s1

    data_loss = torch.sqrt(data_loss)

    return data_loss


if __name__ == '__main__':
    """ dataset """
    ds_dir = 'datasets/geoPose3K_final_publish/'
    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])  # todo fix broadcasting error
                                         ])

    ds = GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform, verbose=False)
    batch_size = 4
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4, )  # collate_fn=ds.collate

    """ model """
    megadepth_checkpoints_path = './megadepth/checkpoints/'

    with patch.object(sys, 'argv', ['/content/geopose/train.py']):
      opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)
    # model = HGModel(opt)
    """ Training """
    # torch.autograd.set_detect_anomaly(True)
    model.netG.train()

    optimizer = torch.optim.Adam(model.netG.parameters(), lr=opt.lr * 10, betas=(opt.beta1, 0.999))

    epochs = 20
    i = 0
    running_loss = 0.0
    loss_history = []

    # with torch.no_grad():  # no training - just evaluate loss
    try:
        for epoch in range(epochs):
            print("epoch: ", epoch)
            for i, batch in enumerate(loader):
                imgs = batch['img'].type(torch.FloatTensor)
                depths = batch['depth']
                masks = batch['mask']
                paths = batch['path']

                # whole batch prediction
                # preds = model.netG.forward(imgs).cpu()

                for sample in range(batch_size):
                    img = imgs[sample]
                    depth = depths[sample]
                    mask = masks[sample]
                    path = paths[sample]

                    i += 1

                    optimizer.zero_grad()
                    img = torch.unsqueeze(img, dim=0)

                    # prediction for single sample
                    pred = model.netG.forward(img).cpu()

                    #pridane pre log
                    pred = torch.squeeze(torch.exp(pred), dim=0)
                    pred_t = torch.log(pred + 2)
                    depth_t = torch.log(depth + 2)
                    loss = rmse_loss(pred_t, depth_t, mask)

                    #loss = rmse_loss(pred, depth, mask)
                    print(loss.item())
                    loss_history.append(loss.item())

                    loss.backward()
                    optimizer.step()

    except KeyboardInterrupt:
        print('stopped training')

    plt.plot(loss_history)
    plt.title('Training loss')
    plt.show()
