import numpy as np


def cap_array_generator(n=12, radix=2, mismatch=0.01, structure='conventional'):
    """
    generates an array of capacitors and computes the binary weights in the dac. there are three
    different structures from which to choose. pay attention that in the returned tuple of capacitor array
    and weights array. the shape of the arrays can be two dimensional or one dimensional according to the structures.
    :param n: number of bits
    :param radix: the radix
    :param mismatch: mismatch of the capacitors
    :param structure: 'conventional': the conventional structure of capacitor divider. the amplifier is single ended.
                      'differential': two capacitor dividers are connected to the positive and negative input of op amp.
                                      The amplifier is single ended. The states of the switches in positive array is
                                      complementary to that in negative array.
                      'split': an attenuator capacitor is placed between the LSB and MSB capacitor array.
    :return: a tuple of capacitor array and weights array.
                    if 'conventional': shape of capacitor array:(n,) , MSB to LSB,
                                       shape of weights array: (n,) ,  MSB to LSB.
                    if 'differential': shape of capacitor array:(2,n+1), MSB to LSB,
                                       shape of weights array: (2,n) , MSB to LSB
                    if 'split':        shape of capacitor array: (n+2,) , LSB to MSB
                                       shape of weights array: (n,) , MSB to LSB
    """
    if structure == 'conventional':
        cap_exp = np.concatenate(([0], np.arange(n)), axis=0)  # exponential of capacitance array
        cap_array = []
        # print('cap_exponential',cap_exp)
        for i in cap_exp:
            cap_i = np.random.normal(radix ** i, mismatch * np.sqrt(radix ** i))  # good case
            cap_array += [cap_i]
        cap_sum = np.sum(cap_array)
        #  reserve the cap_array and abandon the last element
        weights = (np.flip(cap_array, -1)[:-1]) / cap_sum  # binary weights
        return cap_array,weights
    elif structure == 'differential':
        cap_exp = np.concatenate(([0], np.arange(n)), axis=0)  # exponential of capacitance array
        cap_array = np.array([[],[]])
        for i in cap_exp:
            cap_i = np.random.normal(radix ** i, mismatch * np.sqrt(radix ** i),size=(2,1))  # good case
            cap_array = np.hstack((cap_array,cap_i))    # get an (2,n+1) array
        cap_sum = np.sum(cap_array,axis=-1)[:, np.newaxis]  # in order to use broadcasting, get an (2,1) array
        weights = (np.flip(cap_array,-1)[:, :-1]) / cap_sum  # get an (2,n) array
        return cap_array, weights     # cap_array shape(2,n+1); weights shape (2,n)
    elif structure == 'split':
        cap_exp = np.concatenate(([0],np.arange(n/2)),axis=0)
        cap_array = np.array([[],[]])
        for i in cap_exp:
            cap_i = np.random.normal(radix ** i, mismatch * np.sqrt(radix ** i),size=(2,1)) # good case
            cap_array = np.hstack((cap_array,cap_i))   # get an (2,n/2) array
        cap_array_lsb = cap_array[0][:]
        cap_array_msb = cap_array[1][1:]  # MSB array has no dummy capacitor , shape(n/2,)
        cap_sum_lsb = np.sum(cap_array_lsb)
        cap_sum_msb = np.sum(cap_array_msb)
        cap_attenuator = 1  # ideally it should be cap_sum_lsb/cap_sum_msb, but here we set it to 1 directly

        # the series of attenuator capacitor and entire MSB array
        cap_sum_MA = cap_attenuator * cap_sum_msb/(cap_attenuator + cap_sum_msb)
        # the series of attenuator capacitor and entire LSB array
        cap_sum_LA = cap_attenuator * cap_sum_lsb/(cap_attenuator + cap_sum_lsb)

        # attention: the location of positive input of the amplifier is between attenuator capacitor and MSB array
        # so here we need to multiply with an extra term 'cap_attenuator/(cap_attenuator+cap_sum_msb)'
        weights_lsb = (np.flip(cap_array_lsb,-1)[:-1])/(
                cap_sum_lsb + cap_sum_MA) * (cap_attenuator/(cap_attenuator+cap_sum_msb))
        weights_msb = (np.flip(cap_array_msb,-1))/(cap_sum_msb + cap_sum_LA)
        weights = np.concatenate((weights_msb,weights_lsb))

        # attention: in the following step, the concatenated array is LSB-Array + attenuator + MSB-Array,
        # in which the position of MSB and LSB are exchanged if comparing with other structures.
        # However in the weights array, the first element corresponds to the MSB and the last element corresponds
        # to the LSB, which accords with the other structures.
        cap_array = np.concatenate((cap_array_lsb,[cap_attenuator],cap_array_msb))
        return cap_array, weights
