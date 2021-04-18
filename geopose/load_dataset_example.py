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

try:
    from geopose.train import rmse_loss
    from geopose.dataset import GeoPoseDataset, clear_dataset_dir, rotate_images
except ModuleNotFoundError:
    from dataset import GeoPoseDataset, clear_dataset_dir, rotate_images
    from train import rmse_loss


def iter_few(ds):
    cnt = 0
    for _ in ds:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt)
        if cnt == 120:
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

    ds_dir = 'datasets/geoPose3K_mini/'

    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])
                                         ])

    ds = GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform, verbose=False)

    # batch size and num_workers are chosen arbitrarily, try your own ideas
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate)

    timing(iter_few)(loader)

    """
    100 batches * 4 samples, 4 workers => 18s, 
    ..?
    120 batches * 4 samples, 4 workers => 147s, 
    100 batches * 8 samples, 4 workers => 145s
    
    """

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

    for img, depth, mask, path in ds[0:3]:

        f, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        ax[1].imshow(depth)
        ax[2].imshow(mask)
        f.show()
    if False:

            plt.imshow(pred - depth)
            plt.colorbar()
            plt.show()

            depth = resize(depth, pred.shape)

            log_gt = torch.Tensor(np.log(depth + 2))
            log_prediction_d = torch.Tensor(np.log(depth + 2000))

            print(rmse_loss(log_prediction_d, log_gt))

    if False:
        # playing around with RMSE
        img, depth, mask, path = ds[0]
        depth += 2

        # pretend noisier prediction is GT
        depth_fake_gt = depth.copy() - 10 + np.random.random(np.prod(depth.shape)).reshape((depth.shape))

        # mask = np.zeros_like(depth) + 1  # ones-mask

        log_gt = torch.Tensor(depth)
        mask = torch.Tensor(mask)
        # log_prediction_d = torch.Tensor(pred)


    # img, depth, path = ds[0]


