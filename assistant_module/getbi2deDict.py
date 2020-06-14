import numpy as np


def getbi2deDict(n):
    """
    get the binary representation of a n-bit range to decimal number dictionary. e.g {'001':1,'010':2,...}
    :return: a dictionary
    """
    bi2deDict = {}
    decimal = np.arange(2 ** n)
    for x in decimal:
        binary = np.binary_repr(x, n)
        bi2deDict.update({binary: x})
    return bi2deDict