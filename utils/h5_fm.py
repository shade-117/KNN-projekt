import h5py
import matplotlib.pylab as plt
import numpy as np

def open_h5(path):
    filename = path

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        return data



if __name__ == '__main__':
    # res = open_pfm("distance_crop.pfm")
    res = open_h5("data/5008984_74a994ce1c_o.h5")
    plt.imshow(res)
    plt.gca().invert_yaxis()
    plt.show()
