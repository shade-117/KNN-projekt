import math

import numpy as np


def running_mean(arr, window_size, pad_start=False):
    """Get running mean of `arr` over past `n` elements

    inspired by:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    """
    if window_size > len(arr):
        window_size = len(arr)

    cumsum = np.cumsum(np.insert(arr, 0, 0))
    res = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    if pad_start:
        res = np.r_[np.repeat(res[0], window_size), res]
    return res


def timing(f):
    """Time a function execution - as a decorator or timing(foo)(args)

    Taken from
    https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

    :param f:
    :return:
    """

    from functools import wraps
    from time import time

    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'func:{f.__name__!r} args:[{args!r}, {kwargs!r}] \ntook: {end - start:2.4f} sec')
        return result

    return wrap


def inpaint_nan(image, row, col, radius=2):
    """
    Demo: https://colab.research.google.com/drive/1TVjaB7kJMN-jmhrryJOZ7Lggp6Xri1l0?usp=sharing

    :param row: target nan pixel row
    :param col: target nan pixel column
    :param radius: local area size --> filter_size = 1 + 2 * k
    :param image: input image
    :return: local nanmean value for target pixel
    """
    pixel_sum = 0
    pixel_count = 0
    filter_size = 1 + 2 * radius
    for i in range(0, filter_size):
        for j in range(0, filter_size):
            # Out of boundary pixels are skipped.
            if row - radius + i < 0 or col - radius + j < 0 \
                    or row - radius + i > image.shape[0] - 1 or col - radius + j > image.shape[1] - 1:
                continue
            else:
                pixel = image[row - radius + i][col - radius + j]
                if math.isnan(pixel):
                    continue
                else:
                    pixel_sum += pixel
                    pixel_count += 1
    return pixel_sum / pixel_count


def inpaint_nan_numpy(k, image, row, col):
    """
    Demo: https://colab.research.google.com/drive/1TVjaB7kJMN-jmhrryJOZ7Lggp6Xri1l0?usp=sharing

    todo make it work on a connected component (blob) instead of a single nan pixel

    :param row: target nan pixel row
    :param col: target nan pixel column
    :param k: local area size --> filter_size = 1 + 2 * radius
    :param image: input image
    :return: local nanmean value for target pixel
    """

    row = max(row, k)
    row = min(row, image.shape[1] - k - 1)

    col = max(col, k)
    col = min(col, image.shape[1] - k - 1)

    # plt.imshow(image[row - radius:row + radius + 1, col - radius: col + radius + 1].copy())
    # plt.show()
    return np.nanmean(image[row - k:row + k + 1, col - k: col + k + 1])
