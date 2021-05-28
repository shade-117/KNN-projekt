import sys
import time
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from skimage import io
from skimage.transform import resize
import torch
from torchvision import transforms
import seaborn as sns

from geopose.model.builder import Hourglass
from geopose.losses import rmse_loss, gradient_loss
import geopose.dataset as dataset

from utils.process_images import get_sky_mask, transform_image_for_megadepth, megadepth_predict, \
    transform_image_for_semseg, semseg_predict, apply_sky_mask
from utils.semseg import visualize_result


if __name__ == '__main__':
    """
    TODO:
    pip install seaborn, add to requirements?
    
    plot depth histogram for 3k and 17k
    
    plot fov histogram for 3k and 17k
        make it normalized by total count
        
    najít si graf modelu
    
    
    """


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

    uniq_depths = defaultdict()
    fovs = []

    avg_depth = np.zeros((input_width, input_height))

    with torch.no_grad():
        for i, sample in enumerate(ds):
            start = time.time()
            # input_image = sample['img'].cuda()
            mask_img = sample['mask']
            depth_img = sample['depth']
            dir_path = sample['path']
            fov = float(sample['fov'])

            if mask_img.sum() == 0:
                print('Zero sky mask - fishy')

            avg_depth += depth_img

            vals, counts = np.unique(depth_img, return_counts=True)
            for idx, v in enumerate(vals):
                uniq_depths[v] += counts[idx]

            fovs.append(fov)

    depths_val = np.array(list(uniq_depths.keys()))
    depths_cnt = np.array(list(uniq_depths.values()))

    plt.hist(x=depths_val, data=depths_cnt, bins=2000)
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

