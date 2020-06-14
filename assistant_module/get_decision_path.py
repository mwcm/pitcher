import numpy as np
from assistant_module.bin_array import bin_array


def get_decision_path(n):
    """
    get a array of decision path of the full decision tree.
    :param n: depth of the decision tree, it is equivalent to the resolution of
    the DAC.
    :return: A two-dimensional array,each row of which represents the decision path
    of a possible decision level ( a odd decimal integer).
    """
    # n = self.n # depth of the decision tree
    # possible decision level before the last comparision
    code_decimal = np.arange(1, 2**n, 2)
    code_binary = bin_array(code_decimal, n)  # binary digits, shape (len(code_decimal),n)
    # store the decision thresholds generated in each conversion
    decision_path = np.zeros((len(code_decimal), n))
    for i in range(len(code_decimal)):
        code_i = code_decimal[i]
        delta = np.array([2**i for i in range(n-1)])
        D = code_binary[i]
        decision_path[i, -1] = code_i
        decision_path[i, 0] = 2**(n-1)
        for j in range(n-2, 0, -1):
            decision_path[i, j] = decision_path[i, j+1] + (-1)**(2-D[j])*delta[n-2-j]
    return decision_path