import h5py
import matplotlib.pylab as plt
import numpy as np

from utils.pfm import open_pfm


def open_h5(path):
    filename = path

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        depth_key = next(iter(f.keys()))

        # Get the data
        data = np.array(f[depth_key])
        return data


if __name__ == '__main__':
    pfm = open_pfm("data/distance_crop.pfm")
    res = open_h5("data/5008984_74a994ce1c_o.h5")
    print(np.sum(pfm == np.min(pfm)))

    plt.imshow(res)
    plt.colorbar()
    plt.show()

    plt.imshow(pfm)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
