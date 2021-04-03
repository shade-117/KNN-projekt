import matplotlib as mpl
mpl.use('module://backend_interagg')  # fix for PyCharm: show plots in SciView (Petr)
import matplotlib.pyplot as plt

import dataset

if __name__ == '__main__':
    # make a symlink to the dataset or put it into main project folder:
    # ln -s {{path/to/file_or_directory}} {{path/to/symlink}}

    ds_dir = '.../datasets/geoPose3K_final_publish/'

    dataset.clear_dataset_dir(ds_dir)

    ds = dataset.GeoPoseDataset(data_dir=ds_dir)

    # for sample in d:
    if True:
        sample = next(iter(ds))

        f, ax = plt.subplots(1, 3)
        ax[0].imshow(sample['img'])
        ax[1].imshow(sample['depth1'])
        ax[2].imshow(sample['depth2'])
        f.show()
