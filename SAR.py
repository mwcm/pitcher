# -*- coding: utf-8 -*-
# from https://github.com/arutema47/sar-adc

import numpy as np

##########
# written by Ken Yoshioka 1/29/2018
# vectorized sar adc model

# adcin = input signal of the ADC
# # min=-0.5, max=0.5
# # please normalize!!

# bit = number of bits in SAR ADC
# ncomp = noise of the comparator
# ndac = noise of the c-dac
# nsamp = sampling kT/C noise
# radix = radix of the C-DAC


class SAR:
    def __init__(self, bit, ncomp, ndac, nsamp, radix):
        self.ncomp = ncomp
        self.ndac = ndac
        self.nsamp = nsamp
        self.bit = bit
        self.radix = radix
        self.cdac = self.dac()

    def comp(self, compin):
        # note that comparator output suffers from noise of comp and dac
        comptemp = compin + np.random.randn(compin.shape[0])*self.ncomp + np.random.randn(compin.shape[0])*self.ndac

        # comp function in vectors
        out = np.maximum(comptemp*10E6, -1)
        out = np.minimum(out, 1)
        return(out)

    def dac(self):
        cdac = np.zeros((self.bit, 1))
        for i in range(self.bit):
            cdac[i] = np.power(self.radix, (self.bit-1-i))
        cdac = cdac/(sum(cdac)+1)  # normalize to full scale = 1
        # mismatches are tbd
        return(cdac)

    def sarloop(self, adcin):
        # add sampling noise to input first
        adcin += np.random.randn(adcin.shape[0]) * self.nsamp
        adcout = np.zeros_like(adcin)

        # loop for sar cycles
        for cyloop in range(self.bit):
            compout = self.comp(adcin)
            adcin += compout * (-1) * self.cdac[cyloop]  # update cdac output
            adcout += np.power(self.radix, self.bit-1-cyloop)*np.maximum(compout, 0)
            print(cyloop)
        return(adcout)


def normalize_input(inp):
    center = np.mean(inp)
    out = inp - center
    maxbin = np.max(out) * 2
    out = out / maxbin
    return out, center, maxbin

