import contextlib
import os
import shutil
import pathlib
import gzip
from collections.abc import Iterable

import numpy as np
import torch
import imageio  # read PFM

imageio.plugins.freeimage.download()  # download Freelibs for reading PFM files


class GeoPoseDataset(torch.utils.data.Dataset):
    # self.camera_csv = None
    def __init__(self, data_dir='../datasets/geoPose3K_final_publish/'):
        super(GeoPoseDataset).__init__()
        # self.dir = data_dir
        # self.transform = transform
        assert os.path.isdir(data_dir), \
            'Dataset could not find dir "{}"'.format(data_dir)

        self.img_paths = []
        self.depth_paths = []

        for curr_dir in os.listdir(data_dir):
            img_path = os.path.join(data_dir, curr_dir, 'photo.jpeg')
            # depth1_path = os.path.join(data_dir, curr_dir, 'cyl_distance_crop.pfm.gz')
            depth_path = os.path.join(data_dir, curr_dir, 'pinhole_distance_crop.pfm.gz')

            if not (os.path.isfile(img_path) and
                    os.path.isfile(depth_path)):
                continue

            self.img_paths.append(img_path)
            self.depth_paths.append(depth_path)

        if len(self.img_paths) != len(self.depth_paths):
            print("image and depth lists length does not match {} x {}"
                  .format(len(self.img_paths), len(self.depth_paths)))

        print('dataset-length:', len(self.img_paths))  # debug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return (self.__getitem__(i) for i in idx)

        with gzip.open(self.depth_paths[idx], 'r') as depth_archive:
            depth_img = np.flipud(imageio.imread(depth_archive.read(), 'pfm'))

        base_img = imageio.imread(self.img_paths[idx], 'jpeg')

        sample = {
            'depth': depth_img,
            'img': base_img,
            'path': self.img_paths[idx]
        }

        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def clear_dataset_dir(dataset_dir):
    """
    Remove unnecessary files
    Rename ground-truth photo names to jpeg

    Idempotent but you only need to run this once
    """

    old_cwd = os.getcwd()
    try:
        print('clearing dataset directory')

        assert os.path.isdir(dataset_dir), \
            "Invalid directory for clearing dataset"

        os.chdir(dataset_dir)

        depth_gz = 'distance_crop.pfm.gz'

        for curr in os.listdir():
            if not os.path.isdir(curr):
                continue

            pin = curr + os.sep + 'pinhole' + os.sep + depth_gz  # second depth file
            if os.path.exists(pin):
                shutil.move(pin, os.path.join(curr, 'pinhole_' + depth_gz))

            photo = curr + os.sep + 'photo'
            if os.path.exists(photo + '.jpg'):
                os.rename(photo + '.jpg', photo + '.jpeg')

            to_remove = os.listdir(curr)
            to_remove.remove('pinhole_' + depth_gz)
            to_remove.remove('photo.jpeg')
            try:
                to_remove.remove('depth_map_no_sky.npy')
                to_remove.remove('depth_map.npy')
            except ValueError:
                # they were not generated
                ...

            for r in to_remove:
                r_full = curr + os.sep + r
                if os.path.isdir(r_full):
                    shutil.rmtree(r_full)
                else:
                    os.remove(r_full)
    finally:
        os.chdir(old_cwd)
