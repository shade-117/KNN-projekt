# stlib
import contextlib
import shutil
import gzip
from collections.abc import Iterable
import os
import pathlib

# external
import numpy as np
import torch
import imageio  # read PFM
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.transform import resize
from torchvision import transforms
from scipy.ndimage import rotate
from turbojpeg import TurboJPEG


imageio.plugins.freeimage.download()  # download Freelibs for reading PFM files


class GeoPoseDataset(torch.utils.data.Dataset):
    """GeoPose3k as a dataset of dictionaries {path, image, depth}"""

    def __init__(self, ds_dir='datasets/geoPose3K_final_publish/', transforms=None, verbose=False):
        super(GeoPoseDataset).__init__()
        assert os.path.isdir(ds_dir), \
            'Dataset could not find dir "{}"'.format(ds_dir)

        self.verbose = verbose
        self.img_paths = []
        self.depth_paths = []
        self.transforms = transforms
        self.jpeg_reader = TurboJPEG()

        # list only directories, sorting not really necessary
        listed_data_dir = [d for d in sorted(os.listdir(ds_dir)) if os.path.isdir(os.path.join(ds_dir, d))]

        for curr_dir in listed_data_dir:
            img_path = os.path.join(ds_dir, curr_dir, 'photo.jpeg')
            depth_path = os.path.join(ds_dir, curr_dir, 'distance_crop.pfm')

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

        depth_img = imageio.imread(self.depth_paths[idx], format='pfm')
        depth_img = np.flipud(np.array(depth_img)).copy()

        indices = np.argwhere(np.isnan(depth_img))
        mean_depth = np.nanmean(depth_img)
        for ind_nan in indices:
            # stupid indexing but it works
            depth_img[ind_nan[0], ind_nan[1]] = mean_depth

        if self.verbose and len(indices) > 0:
            print('NaN x{} in {}'.format(len(indices), self.img_paths[idx]))

        # before: 88ms
        # base_img = imageio.imread(self.img_paths[idx], format='jpeg')

        with open(self.img_paths[idx], 'rb') as photo_jpeg:
            # now: 46ms
            base_img = self.jpeg_reader.decode(photo_jpeg.read(), 0)  # 0 == RGB
        base_img = np.array(base_img)

        if self.transforms is not None:
            base_img = self.transforms(base_img)
            depth_img = self.transforms(depth_img)

        # sample = {
        #     'depth': depth_img,
        #     'img': base_img,
        #     'path': os.path.dirname(self.img_paths[idx])  # unused
        # }
        # return sample

        return base_img, depth_img, os.path.dirname(self.img_paths[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def collate(data):
        return tuple(zip(*data))


def clear_dataset_dir(ds_dir, resize_dim=None):
    """
    Remove unnecessary files from dataset
    Rename ground-truth photo names to jpeg
    Crop images to dim x dim, if it is set

    Destructive, run only once
    """

    old_cwd = os.getcwd()
    try:

        assert os.path.isdir(ds_dir), \
            "Invalid directory for clearing dataset"

        print('clearing dataset directory started')

        os.chdir(ds_dir)

        is_cleared_file = 'is_cleared.txt'

        if os.path.isfile(is_cleared_file):
            print('dataset directory has already been cleared, skipping')
            return

        depth_pfm = 'distance_crop.pfm'  # depth ground-truth
        mask_png = 'labels_crop.png'  # image segmentation ground-truth

        for curr in os.listdir():
            if not os.path.isdir(curr):
                continue

            # depth map
            pin_path = os.path.join(curr, 'pinhole', depth_pfm + '.gz')  # second depth file
            if os.path.exists(pin_path):
                shutil.move(pin_path, os.path.join(curr, 'pinhole_' + depth_pfm + '.gz'))

            # unzip archive
            pin_path = os.path.join(curr, 'pinhole_' + depth_pfm + '.gz')
            if os.path.isfile(pin_path):
                with gzip.open(pin_path, 'rb') as depth_archive:
                    with open(os.path.join(curr, depth_pfm), 'wb') as depth_archive_content:
                        shutil.copyfileobj(depth_archive, depth_archive_content)

            # base image
            photo_path = os.path.join(curr, 'photo')
            if os.path.exists(photo_path + '.jpg'):
                os.rename(photo_path + '.jpg', photo_path + '.jpeg')

            # segmentation map
            seg_path = os.path.join(curr, 'pinhole', mask_png)
            if os.path.exists(seg_path):
                shutil.move(seg_path, os.path.join(curr, 'pinhole_' + mask_png))

            to_remove = os.listdir(curr)

            # Remove everything except for:
            do_not_remove = [depth_pfm,
                             'pinhole_' + mask_png,
                             'photo.jpeg',
                             'photo.jpeg',
                             'info.txt',
                             'depth_map_no_sky.npy',
                             'depth_map.npy']

            for file in do_not_remove:
                with contextlib.suppress(ValueError):
                    to_remove.remove(file)

            for r in to_remove:
                r_full = os.path.join(curr, r)
                if os.path.isdir(r_full):
                    shutil.rmtree(r_full)
                else:
                    os.remove(r_full)

        # log that dataset has been cleared
        with open(is_cleared_file, 'w+') as f:
            f.write('True')

    except Exception as ex:
        print(ex)
    finally:
        os.chdir(old_cwd)
    print('clearing dataset directory finished')


def copy_images_out(ds_dir='datasets/geoPose3K_final_publish/', dir_to='../geoPose3K_photos_merged/'):
    """Copy all dataset images to a separate folder"""
    old_cwd = os.getcwd()
    try:
        os.makedirs(dir_to, exist_ok=True)

        for curr in os.listdir(ds_dir):
            if not os.path.isdir(curr):
                continue

            photo = os.path.join(curr, 'photo')
            if os.path.exists(photo + '.jpg'):
                os.rename(photo + '.jpg', photo + '.jpeg')

            shutil.copy(photo + '.jpeg', f'../photos/{curr}.jpeg')
    except Exception as ex:
        print(ex)
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

        is_rotated_file = 'is_rotated.txt'

        if os.path.isfile(is_rotated_file):
            print('dataset rotations have already been fixed, skipping')
            return

        if show_cv:
            cv.namedWindow('a')

        for curr in os.listdir():
            if not os.path.isdir(curr):
                continue
            photo = os.path.join(curr, 'photo.jpeg')

            if curr in rotations:
                img = imageio.imread(photo, format='jpeg')
                rot_param = rotations[curr] // 90
                img_rot = rotate(img, rotations[curr])
                imageio.imwrite(photo, img_rot, format='jpeg')

                if show_plt:
                    f, ax = plt.subplots(1, 2)
                    ax[0].imshow(img)
                    ax[1].imshow(img_rot)
                    plt.title('{}\n{}'.format(rot_param, curr))
                    plt.show()

                if show_cv:  # unused - show images (square-croped) orig and rotated
                    smaller_axis = min(img.shape[0], img.shape[1])
                    img = img[:smaller_axis, :smaller_axis]
                    cv.imshow('a', np.hstack([img, img_rot]))
                    cv.waitKey(0)

        if show_cv:
            cv.destroyAllWindows()  # unused

        with open(is_rotated_file, 'w+') as f:
            f.write('True')

    except Exception as ex:
        print(ex)
    finally:
        os.chdir(old_cwd)
