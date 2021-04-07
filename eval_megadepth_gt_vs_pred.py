import geopose.dataset as dataset
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


input_height = 384
input_width = 512
ds_dir = './datasets/geoPose3K_final_publish/'
# dataset.clear_dataset_dir(ds_dir)
ds = dataset.GeoPoseDataset(ds_dir=ds_dir)

for d in ds:
    # d = ds[2511]
    input_image, depth_img, dir_path = d
    gt_resized = np.float32(resize(depth_img, (input_height, input_width), preserve_range=True))

    mask = np.isnan(gt_resized)
    gt_resized = np.where(mask, gt_resized[gt_resized > 0].mean(), gt_resized)
    pred = np.load(dir_path + '/depth_map_no_sky.npy')
    log_pred = np.log((pred + 2))
    log_gt = np.log(gt_resized + 2)
    res = log_gt - log_pred
    print(np.mean(res))
    print(dir_path)
    # print(np.min(res))
    # print(np.min(pred))
    # print(np.min(gt_resized))
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(gt_resized)
    ax[1].imshow(pred)
    f.show()
    plt.show()
