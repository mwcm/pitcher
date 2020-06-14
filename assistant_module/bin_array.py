import numpy as np


def bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int64's.
    if arr is a single integer, the shape of returned array would be (1,m) (2-dimensional array)
    if arr is a ndarray-like sequence with shape(n,), the shape of returned array would be (n,m) (2-dimensional array)
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    arr = np.asarray(arr)
    if arr.shape ==():
        arr = np.asarray([arr])
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int64)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., bit_ix] = fetch_bit_func(strs).astype("int64")

    return ret