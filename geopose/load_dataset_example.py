from collections.abc import Iterable
import gzip
import os
import pathlib
import shutil

import cv2 as cv
import imageio  # read PFM
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from skimage.transform import resize

from geopose.dataset import GeoPoseDataset, clear_dataset_dir, rotate_images


def iter_ds(ds):
    cnt = 0
    for sample in ds:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt)
        if cnt == 100:
            break


def timing(f):
    """Time a function execution - as a decorator or timing(foo)(args)

    Taken from
    https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

    :param f:
    :return:
    """

    from functools import wraps
    from time import time

    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'func:{f.__name__!r} args:[{args!r}, {kwargs!r}] \ntook: {end - start:2.4f} sec')
        return result

    return wrap


if __name__ == '__main__':
    # working folder 'KNN-projekt' assumed

    # make a symlink to the dataset or put it into main project folder:
    # ln -s {{path/to/file_or_directory}} {{path/to/symlink}}

    ds_dir = 'datasets/geoPose3K_final_publish/'

    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((384, 512)),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    ds = GeoPoseDataset(data_dir=ds_dir, transforms=data_transform)

    # batch size and num_workers are chosen arbitrarily, try your own ideas
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate)

    if False:
        # see how long it takes to load data
        # can be run with ds (returning samples) or loader (returning batches)
        iter_ds(loader)

        # single element directly from dataset
        a = ds[0]

        # single batch through dataloader
        b = next(iter(loader))

        # dataset also supports slicing
        sample = ds[0:3]

    if False:
        for img, depth in ds[0:3]:

            f, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[1].imshow(depth)
            f.show()

    if True:
        # playing around with RMSE
        img, depth = ds[0]
        depth += 20

        # pretend noisier prediction is GT
        depth_fake_gt = depth.copy() - 10 + np.random.random(np.prod(depth.shape)).reshape((depth.shape))

        mask = np.zeros_like(depth) + 1  # ones-mask

        log_gt = torch.Tensor(depth_fake_gt)
        log_prediction_d = torch.Tensor(depth)

        mask = torch.Tensor(mask)


def rmse_loss(log_prediction_d, mask, log_gt):
    # from rmse_error_main.py
    n = torch.sum(mask)

    log_d_diff = log_prediction_d - log_gt
    log_d_diff = torch.mul(log_d_diff, mask)

    s1 = torch.sum(torch.pow(log_d_diff, 2)) / n
    s2 = torch.pow(torch.sum(log_d_diff), 2) / (n * n)

    data_loss = s1 - s2

    data_loss = torch.sqrt(data_loss)

    return data_loss
