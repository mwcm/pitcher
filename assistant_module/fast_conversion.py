import numpy as np
from assistant_module.get_decision_lvls import get_decision_lvls


def fast_conversion(analog_samples, weights, n, vref):
    """
    uses the fast conversion algorithm to convert an array of analog_samples into
    decimal digital values.
    :param analog_samples: a array with one dimension
    :param weights: binary weights of adc
    :param n: number of bits
    :param vref: reference voltage of adc
    :return: a array of decimal integers,whose number of dimension is 1 and length
            equal to the length of analog_samples
    """
    # convert analog input to array and add one dimension
    # use asarray method to handle the case that analog_samples is a single value.
    analog_samples = np.asarray(analog_samples)[:, np.newaxis]    # shape(M,1)
    decision_lvls = get_decision_lvls(weights, n, vref)[np.newaxis, :]    # shape(1,N)
    # use numpy broadcasting to compare two matrix element wise
    relation_matrix = np.asarray(np.greater_equal(analog_samples, decision_lvls), dtype=np.int64)
    # sum each row and minus 1 getting a array with shape(M,)
    conversion_result = np.sum(relation_matrix, axis=-1) - 1
    return conversion_result
