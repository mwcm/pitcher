'''
Copyright 2012 Stefano D'Angelo <zanga.mail@gmail.com>
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.
THIS SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/
'''

from numpy import tanh


# Based on the Improved Model from ddiakopoulos' MoogLadders repo
# https://github.com/ddiakopoulos/MoogLadders

# The Improved Model itself was created by D'Angelo & Valimaki in 2013's 
# "An Improved Virtual Analog Model of the Moog Ladder Filter" 
# (https://raw.githubusercontent.com/ddiakopoulos/MoogLadders/master/research/DAngeloValimaki.pdf)


# "Two of these filters employ the SSM-2044 Voltage Controlled Filter (VCF) chip as a 4-pole lowpass with time-varying cutoff frequency."
# VCF functionality only used for outputs 1 & 2 

# could test against slides and match sp-1200's use of ssm2044 for outputs 1&2
#  - Exponential time constant = 0.085 s
#  - Initial Fc = 14150
#  - Final Fc = 1150

# for now just exposing cutoff through pitcher options
# since mostly using for general audio not just kicks


# Thermal voltage (26 miliwatts at room temp)
VT = 0.312
MOOG_PI = 3.14159265358979323846264338327950288

class LadderFilterBase:
	def __init__(self, sample_rate, cutoff=0, resonance=0):
		self.sample_rate = sample_rate
		self.cutoff = cutoff
		# should likely put limits on this (ie 4>res>0)
		self.resonance = resonance
		return 

	def process(self, samples):
		return samples

	def getResonance(self):
		return self.resonance

	def getCutoff(self):
		return self.cutoff

	def setResonance(self, res):
		self.resonance = res

	def setCutoff(self, cutoff):
		self.cutoff = cutoff


class MoogFilter(LadderFilterBase):
	def __init__(self, sample_rate=48000, cutoff=10000, resonance=0.1, drive=1.0):
		self.sample_rate = sample_rate
		self.resonance = resonance
		self.drive = drive
		self.x = 0
		self.g = 0
		self.V = [0,0,0,0]
		self.dV = [0,0,0,0]
		self.tV = [0,0,0,0]
		self.setCutoff(cutoff)
	
	def process(self, samples):
		dV0 = 0
		dV1 = 0
		dV2 = 0
		dV3 = 0

		for i, s in enumerate(samples):
			dV0 = -self.g * (tanh((self.drive * samples[i] + self.resonance * self.V[3]) / (2.0 * VT)) + self.tV[0])
			self.V[0] += (dV0 + self.dV[0]) / (2.0 * self.sample_rate)
			self.dV[0] = dV0
			self.tV[0] = tanh(self.V[0] / (2.0 * VT))
			
			dV1 = self.g * (self.tV[0] - self.tV[1])
			self.V[1] += (dV1 + self.dV[1]) / (2.0 * self.sample_rate)
			self.dV[1] = dV1
			self.tV[1] = tanh(self.V[1] / (2.0 * VT))
			
			dV2 = self.g * (self.tV[1] - self.tV[2])
			self.V[2] += (dV2 + self.dV[2]) / (2.0 * self.sample_rate)
			self.dV[2] = dV2
			self.tV[2] = tanh(self.V[2] / (2.0 * VT))
			
			dV3 = self.g * (self.tV[2] - self.tV[3])
			self.V[3] += (dV3 + self.dV[3]) / (2.0 * self.sample_rate)
			self.dV[3] = dV3
			self.tV[3] = tanh(self.V[3] / (2.0 * VT))
			
			samples[i] = self.V[3]

		return samples
	
	def setCutoff(self, cutoff):
		self.cutoff = cutoff
		self.x = (MOOG_PI * cutoff) / self.sample_rate
		self.g = 4.0 * MOOG_PI * VT * cutoff * (1.0 - self.x) / (1.0 + self.x)
