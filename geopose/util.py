import numpy as np


def running_mean(arr, n, pad_start=False):
    """Get running mean of `arr` over past `n` elements

    inspired by:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    """
    if n > len(arr):
        n = len(arr)

    cumsum = np.cumsum(np.insert(arr, 0, 0))
    res = (cumsum[n:] - cumsum[:-n]) / float(n)
    if pad_start:
        res = np.r_[np.repeat(res[0], n), res]
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
