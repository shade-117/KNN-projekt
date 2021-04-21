# stdlib
from collections.abc import Iterable
from collections.abc import Iterable
from unittest.mock import patch
from unittest.mock import patch
import math
import os
import pathlib
import sys
import time

# external
from scipy import misc
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import cv2 as cv
import imageio  # read PFM
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torchvision.transforms

# local
from geopose.dataset import GeoPoseDataset, clear_dataset_dir, rotate_images
from geopose.util import running_mean
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


def rmse_loss(pred, gt, mask=None, scale_invariant=True):
    # from rmse_error_main.py
    assert gt.shape == pred.shape, \
        'Loss tensor shapes invalid: {} x {}'.format(gt.shape, pred.shape)

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


def gradient_loss(pred, gt, mask):
    # adapted from:
    # https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/models/networks.py#L231
    # ^ origin of different targets_1['mask_X']
    # https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/data/image_folder.py#L268

    def gradient_loss_inner(pred, gt, mask):
        n = torch.sum(mask)
        diff = pred - gt
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[0:-2, :] - diff[2:, :])
        v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, 0:-2] - diff[:, 2:])
        h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        loss = loss / n
        return loss

    pred_div2 = pred[:, ::2, ::2]
    pred_div4 = pred_div2[:, ::2, ::2]
    pred_div8 = pred_div4[:, ::2, ::2]

    # downsampling can be calculated during loading dataset
    # but then we would have to apply the same augmentations to these
    gt_div2 = gt[:, ::2, ::2]
    gt_div4 = gt_div2[:, ::2, ::2]
    gt_div8 = gt_div4[:, ::2, ::2]

    mask_div2 = mask[:, ::2, ::2]
    mask_div4 = mask_div2[:, ::2, ::2]
    mask_div8 = mask_div4[:, ::2, ::2]

    loss = gradient_loss_inner(pred, gt, mask)
    loss += gradient_loss_inner(pred_div2, gt_div2, mask_div2)
    loss += gradient_loss_inner(pred_div4, gt_div4, mask_div4)
    loss += gradient_loss_inner(pred_div8, gt_div8, mask_div8)

    return loss


# data x grad loss weighted 2:1 in Fickinger's repo
# https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/models/networks.py#L121

if __name__ == '__main__':

    try:
        from IPython import get_ipython

        running_in_colab = 'google.colab' in str(get_ipython())
    except:
        running_in_colab = False

    if running_in_colab:
        saved_weights_dir = '/content'
    else:
        saved_weights_dir = os.path.join('geopose', 'saved_models')
        os.makedirs(saved_weights_dir, exist_ok=True)

    """ dataset """
    ds_dir = os.path.join('datasets', 'geoPose3K_final_publish')

    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    """
    todo:

    training/validation/test split

    gradient loss

    augmentace

    nan nahradit okolními hodnotami

    """

    """ Dataset """
    batch_size = 4

    ds = GeoPoseDataset(ds_dir=ds_dir, transforms=None, verbose=False)

    # split into training & validation (no test for now)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    validation_split = .1
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    loader_kwargs = {'batch_size': batch_size, 'num_workers': 2}
    train_loader = torch.utils.data.DataLoader(ds, sampler=train_sampler, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(ds, sampler=val_sampler, **loader_kwargs)

    """ Model """
    megadepth_checkpoints_path = './megadepth/checkpoints/'

    curr_script_path = os.path.join(os.getcwd(), 'geopose', 'train.py')
    with patch.object(sys, 'argv', [curr_script_path]):
        # fix for colab interpreter arguments
        opt = TrainOptions().parse(quiet=True)  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)
    # model = HGModel(opt)

    """ Training """
    # torch.autograd.set_detect_anomaly(True)  # debugging

    optimizer = torch.optim.Adam(model.netG.parameters(), lr=opt.lr * 100, betas=(opt.beta1, 0.999))
    # optimizer = torch.optim.Adam(model.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    epochs = 50
    epochs_trained = 0

    i = 0
    running_loss = 0.0
    train_loss_history = []
    val_loss_history = []
    scale_invariancy = False
    stop_training = False  # break training loop flag

    # with torch.no_grad():  # enable when not training to evaluate loss only
    for epoch in range(epochs_trained, epochs_trained + epochs):

        model.netG.train()
        print("epoch:", epoch)
        try:
            for i, batch in enumerate(train_loader):
                start = time.time()
                imgs = batch['img'].type(torch.FloatTensor).permute(0, 3, 1, 2)  # from NHWC to NCHW
                # todo imgs transformations could be a part of transforms

                depths = batch['depth'].cuda()
                masks = batch['mask'].cuda()
                paths = batch['path']

                # batch prediction
                preds = model.netG.forward(imgs)
                preds = torch.squeeze(preds, dim=1)

                # # pridane pre logaritmovanie
                # preds = torch.squeeze(torch.exp(preds), dim=0)
                # preds_t = torch.log(preds + 2)
                # depths_t = torch.log(depths + 2)
                # batch_loss = rmse_loss(preds_t, depths_t, masks, scale_invariant=scale_invariancy)

                data_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariancy)
                grad_loss = gradient_loss(preds, depths, masks)
                batch_loss = (data_loss + grad_loss) / batch_size

                train_loss_history.append(batch_loss.item())

                batch_loss.backward()
                optimizer.step()

                print("\t{:>4}/{} : d={:<9.2f} g={:<9.2f} t={:.2f}s ".format(i, len(train_loader), batch_loss.item(), grad_loss.item(), time.time() - start))

        except KeyboardInterrupt:
            print('stopped training')
            # doesn't skip evaluation and saving weights
            stop_training = True

        epoch_mean_loss = np.mean(train_loss_history)
        save_path = f'saved_{epoch}_{epoch_mean_loss:.4f}_net_G.pth'
        save_path = os.path.join(saved_weights_dir, save_path)
        torch.save(model.netG.state_dict(), save_path)

        model.netG.eval()
        with torch.no_grad():
            print('val:')
            for i, batch in enumerate(val_loader):
                start = time.time()
                imgs = batch['img'].type(torch.FloatTensor).permute(0, 3, 1, 2)  # from NHWC to NCHW

                depths = batch['depth'].cuda()
                masks = batch['mask'].cuda()
                paths = batch['path']

                preds = model.netG.forward(imgs)
                preds = torch.squeeze(preds, dim=1)

                data_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariancy)
                grad_loss = gradient_loss(preds, depths, masks)
                batch_loss = (data_loss + grad_loss) / batch_size

                print("\t{:>4}/{} : d={:<9.2f} g={:<9.2f} t={:.2f}s ".format(i + 1, len(val_loader), batch_loss.item(), grad_loss.item(), time.time() - start))

                val_loss_history.append(batch_loss.item())

        if stop_training:
            break

    epochs_trained += len(train_loss_history)

    # Results

    plt.plot(train_loss_history)
    plt.plot(running_mean(train_loss_history, 100, pad_start=True))
    plt.title('Training loss \n(scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    plt.xlabel('batch (size = {}, ds_size = {})'.format(batch_size, dataset_size))
    plt.ylabel('RMSE loss')
    plt.legend(['train', 'train-mean'])
    plt.show()

    plt.plot(val_loss_history)
    plt.title('Validation loss \n(scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    plt.show()

    # todo uncomment in colab for model saving
    # save_path = '/content/saved_net_G.pth'
    # torch.save(model.netG.state_dict(), save_path)
