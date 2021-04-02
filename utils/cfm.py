import re
import sys
from struct import *

import matplotlib.pylab as plt
import numpy as np


def open_pfm(path):
    with open(path, "rb") as f:
        # Enable/disable debug output
        debug = True

        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        pfm_type = f.readline().decode('ascii')
        if "PF" in pfm_type:
            channels = 3
        elif "Pf" in pfm_type:
            channels = 1
        else:
            print("ERROR: Not a valid PFM file", file=sys.stderr)
            sys.exit(1)
        if debug:
            print("DEBUG: channels={0}".format(channels))

        # Line 2: width height
        line = f.readline().decode('ascii')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)
        if debug:
            print("DEBUG: width={0}, height={1}".format(width, height))

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('ascii')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        if debug:
            print("DEBUG: BigEndian={0}".format(BigEndian))

        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)

        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, newshape=(height, width))
        return img


def open_pfm_custom(path):
    with open(path, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        _ = f.readline()
        # Line 2: width height
        _ = f.readline()
        width, height = 1302, 687
        # Line 3: +ve number means big endian, negative means little endian
        _ = f.readline()
        BigEndian = False
        # Slurp all binary data
        channels = 1
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, newshape=(height, width))
        return img


if __name__ == '__main__':
    # res = open_pfm("distance_crop.pfm")
    res = open_pfm_custom("data/distance_crop2.pfm")

    plt.imshow(res)
    plt.gca().invert_yaxis()
    plt.show()
