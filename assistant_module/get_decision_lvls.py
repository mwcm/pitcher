import numpy as np
from assistant_module.bin_array import bin_array


def get_decision_lvls(weights,n,vref):
    """
    computes all the decision levels(also called transition points)
    :param weights: binary weights of DAC
    :param n: number of bits
    :param vref: reference voltage
    :return: a array of decision levels
    """
    binary_codes = bin_array(np.arange(2**n), n)
    decision_lvls = np.inner(binary_codes, weights) * vref
    return decision_lvls
