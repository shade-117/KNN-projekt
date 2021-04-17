import numpy as np
import matplotlib.pyplot as plt
import geopose.dataset as dataset
import os
import random

ds_dir = './datasets/geoPose3K_final_publish/'
dataset.clear_dataset_dir(ds_dir)
ds = dataset.GeoPoseDataset(ds_dir=ds_dir)
nums = random.sample(range(0, len(ds)), 2)
for i in nums:
    print(i)
    input_image, depth_img, mask, dir_path = ds[i]

    dir_path = os.path.dirname(os.path.realpath(dir_path))
    os.path.join(dir_path, 'depth_map.npy')
    megadepth_pred = np.load(os.path.join(dir_path, 'depth_map.npy'))
    megadepth_pred_no_sky = np.load(os.path.join(dir_path, 'depth_map_no_sky.npy'))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.imshow(input_image)
    ax2.imshow(np.log(depth_img + 2))
    ax3.imshow(megadepth_pred)
    ax4.imshow(np.log(megadepth_pred_no_sky + 2))
    fig.show()
    plt.show()
    # fig.savefig('./figs/' + str(i) + '.png', dpi=110)
    # fig.clear()
