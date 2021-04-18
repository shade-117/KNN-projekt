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

    if mask is None:
        mask = torch.zeros(pred.shape) + 1

    n = torch.sum(mask)

    diff = pred - gt
    diff = torch.mul(diff, mask)

    s1 = torch.sum(torch.pow(diff, 2)) / n

    if scale_invariant:
        s2 = torch.pow(torch.sum(diff), 2) / (n * n)
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

    """
    data_transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.CenterCrop((384, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
                             
    Proč nepoužíváme transforms:
    ToTensor()
        o převod na tensor se postará dataloader a s touto transformací
        pak měly tensory vytvořené datasetem + dataloaderem dvě dimenze pro batch
        (1, 1, 3, 384, 512) pro image input (3, 384, 512)
          
    CenterCrop()
        kazilo to mask_sky
        resize na (384, 512) probíhá při načítání vstupních souborů v datasetu
    
    Normalize()
        házelo to broadcasting error
        nevím, co by normalizace udělala s hloubkovou mapou
    """

    """
    todo:
    
    training/validation/test split
    
    gradient loss
    
    augmentace
    
    nan nahradit okolními hodnotami
    
    """

    """ Dataset """
    batch_size = 1
    ds = GeoPoseDataset(ds_dir=ds_dir, transforms=None, verbose=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)

    """ Model """
    megadepth_checkpoints_path = './megadepth/checkpoints/'
    curr_script_path = os.path.join(os.getcwd(), 'geopose', 'train.py')
    with patch.object(sys, 'argv', [curr_script_path]):
        # fix for runnning code in interactive console/colab/notebooks
        opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)
    # model = HGModel(opt)

    """ Training """
    # torch.autograd.set_detect_anomaly(True)  # debugging
    model.netG.train()

    optimizer = torch.optim.Adam(model.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    epochs = 20
    i = 0
    running_loss = 0.0
    loss_history = []
    scale_invariancy = False

    try:
        # with torch.no_grad():  # enable when not training to evaluate loss only
        for epoch in range(epochs):
            print("epoch: ", epoch)
            for i, batch in enumerate(loader):
                imgs = batch['img'].type(torch.FloatTensor).permute(0, 3, 1, 2)  # from NHWC to NCHW
                # todo imgs transformations could be a part of transforms

                depths = batch['depth']
                masks = batch['mask']
                paths = batch['path']

                # batch prediction
                preds = model.netG.forward(imgs).cpu()
                preds = torch.squeeze(preds, dim=1)

                # # pridane pre logaritmovanie
                # pred = torch.squeeze(torch.exp(pred), dim=0)
                # pred_t = torch.log(pred + 2)
                # depth_t = torch.log(depth + 2)
                # loss = rmse_loss(pred_t, depth_t, mask, scale_invariant=False)

                batch_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariancy)
                batch_loss = batch_loss / batch_size

                print(batch_loss.item())
                loss_history.append(batch_loss.item())

                batch_loss.backward()
                optimizer.step()

    except KeyboardInterrupt:
        print('stopped training')

    plt.plot(loss_history)
    plt.title('Training loss \n(scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    plt.xlabel('batch (size={})'.format(batch_size))
    plt.ylabel('RMSE loss')
    plt.show()

    # todo uncomment in colab for model saving
    # save_path = '/content/saved_net_G.pth'
    # torch.save(model.netG.state_dict(), save_path)
