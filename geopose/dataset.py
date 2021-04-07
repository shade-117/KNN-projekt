# stlib
import os
import shutil
import pathlib
import gzip
from collections.abc import Iterable
import os

# external
import numpy as np
import torch
import imageio  # read PFM
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.transform import resize
from torchvision import transforms


imageio.plugins.freeimage.download()  # download Freelibs for reading PFM files


class GeoPoseDataset(torch.utils.data.Dataset):
    """GeoPose3k as a dataset of dictionaries {path, image, depth}"""

    def __init__(self, data_dir='datasets/geoPose3K_final_publish/', transforms=None, verbose=False):
        super(GeoPoseDataset).__init__()
        assert os.path.isdir(data_dir), \
            'Dataset could not find dir "{}"'.format(data_dir)

        self.verbose = verbose
        self.img_paths = []
        self.depth_paths = []
        self.transforms = None

        for curr_dir in os.listdir(data_dir):
            img_path = os.path.join(data_dir, curr_dir, 'photo.jpeg')
            depth_path = os.path.join(data_dir, curr_dir, 'pinhole_distance_crop.pfm.gz')

            if not (os.path.isfile(img_path) and
                    os.path.isfile(depth_path)):
                continue

            self.img_paths.append(img_path)
            self.depth_paths.append(depth_path)

        if len(self.img_paths) != len(self.depth_paths):
            print("image and depth lists length does not match {} x {}"
                  .format(len(self.img_paths), len(self.depth_paths)))

        if self.verbose:
            print('dataset-length:', len(self.img_paths))  # debug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # allow for slicing -> dataset[0:10]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        elif isinstance(idx, Iterable):
            return (self.__getitem__(i) for i in idx)
        elif not isinstance(idx, int):
            return TypeError('Invalid index type')

        with gzip.open(self.depth_paths[idx], 'r') as depth_archive:
            depth_img = depth_archive.read()
            depth_img = imageio.imread(depth_img, format='pfm')
            depth_img = np.flipud(np.array(depth_img)).copy()

        indices = np.argwhere(np.isnan(depth_img))
        mean_depth = np.nanmean(depth_img)
        for ind_nan in indices:
            # stupid indexing but it works
            depth_img[ind_nan[0], ind_nan[1]] = mean_depth

        if self.verbose and len(indices) > 0:
            print('NaN x{} in {}'.format(len(indices), self.img_paths[idx]))

        base_img = np.array(imageio.imread(self.img_paths[idx], format='jpeg'))

        if self.transforms is not None:
            base_img = self.transforms(base_img)
            depth_img = self.transforms(depth_img)

        # sample = {
        #     'depth': depth_img,
        #     'img': base_img,
        # }
        # 'path': os.path.dirname(self.img_paths[idx])  # unused
        # return sample

        return base_img, depth_img

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def collate(data):
        return tuple(zip(*data))


def clear_dataset_dir(dataset_dir, resize_dim=None):
    """
    Remove unnecessary files from dataset
    Rename ground-truth photo names to jpeg
    Crop images to dim x dim, if it is set

    Destructive, run only once
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

            # depth
            pin = curr + os.sep + 'pinhole' + os.sep + depth_gz  # second depth file
            if os.path.exists(pin):
                shutil.move(pin, os.path.join(curr, 'pinhole_' + depth_gz))

            # base image
            photo = curr + os.sep + 'photo'
            if os.path.exists(photo + '.jpg'):
                os.rename(photo + '.jpg', photo + '.jpeg')


            """
            # Unused
            base_img = imageio.imread(photo + '.jpeg', format='jpeg')
            # cropping (optional)
            if resize_dim is not None:
                r = resize_dim // 2
                img_center = base_img.shape[0] // 2, base_img.shape[1] // 2

                assert min(img_center) >= r

                cropped_img = base_img[img_center[0] - r: img_center[0] + r,
                                       img_center[1] - r: img_center[1] + r]

                cropped_path = photo + '_crop.jpeg'
                # if not os.path.isfile(cropped_path):
                imageio.imwrite(cropped_path, cropped_img, format='jpeg')
            """

            to_remove = os.listdir(curr)

            # Remove everything except for:
            to_remove.remove('pinhole_' + depth_gz)
            to_remove.remove('photo.jpeg')
            to_remove.remove('into.txt')
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


def copy_images_out(ds_dir='datasets/geoPose3K_final_publish/', dir_to='../geoPose3K_photos_merged/'):
    """Copy all dataset images to a separate folder"""
    old_cwd = os.getcwd()
    try:
        os.makedirs(dir_to, exist_ok=True)

        for curr in os.listdir(ds_dir):
            if not os.path.isdir(curr):
                continue

            photo = curr + os.sep + 'photo'
            if os.path.exists(photo + '.jpg'):
                os.rename(photo + '.jpg', photo + '.jpeg')

            shutil.copy(photo + '.jpeg', f'../photos/{curr}.jpeg')
    finally:
        os.chdir(old_cwd)


def rotate_images(ds_dir='datasets/geoPose3K_final_publish/', show_cv=False, show_plt=False):
    """Fix wrong image rotations"""

    # This tells the images current rotation. Apply reverse rotation to fix them.
    # If -90, then sky is on the left. Apply rot(-(-90)) to fix.
    rotations = {
        'flickr_sge_11271557385_3cb11b8a3a_3783_68097255@N00': -90,
        'flickr_sge_3116199276_e23b30b95e_3268_31934892@N06': 90,
        'flickr_sge_4850191705_f5cb1f29b6_4082_21862224@N00': 90,
        'flickr_sge_4850826738_a2d7de7331_4073_21862224@N00': 90,
        'flickr_sge_4852649350_ec572e2cbd_4097_20708977@N00_grid_1_0.008_0.008.xml_1_0_0.936035': -90,
        'flickr_sge_5106859656_3ace495217_4110_23707333@N00': -90,
        'flickr_sge_5243148462_dd09e4b77f_o': 0,  # appears as -90 in regular image browser but is ok when loaded here
        'flickr_sge_7165328218_a1bbeabf0a_7211_77846498@N07_grid_1_0.004_0.004.xml_1_0_0.679168': 90,
        'flickr_sge_7838909292_e1328df4bc_8437_78456006@N02': 180,
    }

    old_cwd = os.getcwd()
    try:
        assert os.path.isdir(ds_dir), \
            "Invalid directory for applying rotations"

        os.chdir(ds_dir)

        if show_cv:
            cv.namedWindow('a')

        for curr in os.listdir():
            if not os.path.isdir(curr):
                continue
            photo = curr + os.sep + 'photo.jpeg'

            if curr in rotations:
                img = imageio.imread(photo)

                rot_param = rotations[curr] // 90

                if show_plt:
                    f, ax = plt.subplots(1, 2)
                    ax[0].imshow(img)
                    ax[1].imshow(np.rot90(img, k=rot_param))
                    plt.title('{}\n{}'.format(rot_param, curr))
                    plt.show()

                if show_cv:  # unused - show images (square-croped) orig and rotated
                    smaller_axis = min(img.shape[0], img.shape[1])
                    img = img[:smaller_axis, :smaller_axis]
                    cv.imshow('a', np.hstack([img, np.rot90(img, k=rot_param)]))
                    cv.waitKey(0)

                imageio.imwrite(photo, np.rot90(img, k=rot_param), format='jpeg')

        if show_cv:
            cv.destroyAllWindows()  # unused
    finally:
        os.chdir(old_cwd)
