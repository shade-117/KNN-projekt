import numpy as np
import matplotlib.pyplot as plt
import geopose.dataset as dataset
import os


ds_dir = './datasets/geoPose3K_final_publish_500mb_part/'
dataset.clear_dataset_dir(ds_dir)
ds = dataset.GeoPoseDataset(data_dir=ds_dir)

sample = ds[0]
dir_path = os.path.dirname(os.path.realpath(sample['path']))
os.path.join(dir_path, 'depth_map.npy')
megadepth_pred_backup = np.load(os.path.join(dir_path, 'depth_map.npy'))
megadepth_pred = np.load(os.path.join(dir_path, 'depth_map_no_sky.npy'))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.imshow(sample['img'])
ax2.imshow(sample['depth1'])
ax3.imshow(megadepth_pred_backup)
ax4.imshow(megadepth_pred)
fig.show()
plt.show()