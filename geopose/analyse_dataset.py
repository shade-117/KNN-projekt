import sys
import time
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import seaborn as sns

from geopose.model.builder import Hourglass
from geopose.losses import rmse_loss, gradient_loss
import geopose.dataset as dataset


if __name__ == '__main__':
    """
    TODO:
    
    plot depth histogram for 3k and 17k
    
    plot fov histogram for 3k and 17k
        make it normalized by total count
        
    najít si graf modelu
    
    
    """
    """ Run with KNN-projekt as the working directory. """

    """ Input sizes """
    input_height = 384
    input_width = 512
    # ds_dir = '/storage/brno3-cerit/home/xmojzi08/geoPose3K_final_publish'
    ds_dir = 'datasets/geoPose3K_final_publish/'
    # dataset.clear_dataset_dir(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])  # todo fix broadcasting error
                                         ])

    ds = dataset.GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform)

    # loader_kwargs = {'batch_size': 8, 'num_workers': 4, 'pin_memory': True, 'drop_last': True}
    # loader = torch.utils.data.DataLoader(ds, sampler=None, **loader_kwargs)

    uniq_depths = defaultdict(int)
    fovs = []

    avg_depth = np.zeros((input_height, input_width))

    with torch.no_grad():
        for i, sample in enumerate(ds[:10]):
            if i % 10 == 0:
                print(i)

            start = time.time()
            # input_image = sample['img'].cuda()
            mask_img = sample['mask']
            depth_img = sample['depth']
            dir_path = sample['path']
            fov = float(sample['fov'])

            if mask_img.sum() == 0:
                print('Zero sky mask - fishy')

            avg_depth += depth_img.numpy()[0]

            vals, counts = np.unique(depth_img, return_counts=True)
            for idx, v in enumerate(vals):
                uniq_depths[v] += counts[idx]

            fovs.append(fov)

    depths_val = np.array(list(uniq_depths.keys()))
    depths_cnt = np.array(list(uniq_depths.values()))

    plt.hist(x=depths_val, weights=depths_cnt, bins=50)
    plt.title('Depth values occurences [metres]')

    plt.show()

    # field of view histogram
    plt.hist(fovs)

    plt.title('FOV occurences')  # [rads] ?
    plt.show()

    avg_depth /= len(ds)
    negative_avg_depth_cnt = np.sum(avg_depth < 0)
    if negative_avg_depth_cnt > 0:
        print(f'negative_avg_depth_cnt = {negative_avg_depth_cnt}')

    # normalize avg depth to [0 .. 255]
    avg_depth[avg_depth < 0] = 0
    avg_depth /= np.max(avg_depth)
    avg_depth *= 255
    avg_depth = avg_depth.astype(np.int32)

    plt.imshow(avg_depth, cmap='viridis')
    plt.title('Average depth map')
    plt.show()

