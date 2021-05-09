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
    from geopose.losses import rmse_loss
    from geopose.dataset import GeoPoseDataset, clear_dataset_dir, rotate_images
except ModuleNotFoundError:
    from dataset import GeoPoseDataset, clear_dataset_dir, rotate_images
    from train import rmse_loss


if __name__ == '__main__':
    # working folder 'KNN-projekt' assumed

    # make a symlink to the dataset or put it into main project folder:
    # ln -s {{path/to/file_or_directory}} {{path/to/symlink}}

    ds_dir = 'datasets/geoPose3K_final_publish/'

    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])
                                         ])

    from geopose.dataset import get_dataset_loaders

    datasets = get_dataset_loaders(ds_dir, batch_size=1, workers=0, validation_split=0.2, shuffle=False, fraction=1.0, random=False)


    if False:
        # single element directly from dataset
        a = ds[0]

        # single batch through dataloader
        b = next(iter(loader))

        # dataset also supports slicing
        sample = ds[0:3]

        for sample in ds:
            continue
            # img, depth, mask, path
            depth = sample['depth']

            # f, ax = plt.subplots(1, 3)
            # ax[0].imshow(img)
            # ax[1].imshow(depth)
            # ax[2].imshow(mask)
            # f.show()

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
        sample = ds[0]
        img = sample['img']
        depth = sample['depth']
        mask = sample['mask']
        path = sample['path']

        # pretend noisier prediction is GT
        depth_fake_gt = depth.copy() - 10 + np.random.random(np.prod(depth.shape)).reshape(depth.shape)

        # mask = np.zeros_like(depth) + 1  # ones-mask

        log_gt = torch.Tensor(depth)
        mask = torch.Tensor(mask)
        # log_prediction_d = torch.Tensor(pred)

    if False:
        # img, depth, path = ds[0]

        # path = 'datasets/geoPose3K_mini/28561570606/distance_crop.pfm'
        path = 'datasets/geoPose3K_final_publish/flickr_sge_10163768706_01e5d4a0a6_o_grid_1_0.004_0.004.xml_1_1_1.1327/distance_crop.pfm'
        depth_img = imageio.imread(path, format='pfm')
        depth_img = np.flipud(np.array(depth_img)).copy()

        dc = depth_img.copy()

        depth_img[np.isnan(depth_img)] = np.nanmean(depth_img)

        # find connected components

        # for every component:
        #   replace by mean of component neighborhood

        plt.imshow(depth_img)
        plt.show()

        np.save('nan3x.npy', depth_img)

