import numpy as np


# TODO: comment and reference


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

	# should likely put limits on this (ie 4>res>0)
	def setResonance(self, res):
		self.resonance = res

	def setCutoff(self, cutoff):
		self.cutoff = cutoff


class MoogFilter(LadderFilterBase):
	# TODO: what should default sample rate be? 44100? 48000?
	# TODO: what should default cutoff be?
	def __init__(self, sample_rate=26040, cutoff=1500, resonance=0.1, drive=1.0):
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
			dV0 = -self.g * (np.tanh((self.drive * samples[i] + self.resonance * self.V[3]) / (2.0 * VT)) + self.tV[0])
			self.V[0] += (dV0 + self.dV[0]) / (2.0 * self.sample_rate)
			self.dV[0] = dV0
			self.tV[0] = np.tanh(self.V[0] / (2.0 * VT))
			
			dV1 = self.g * (self.tV[0] - self.tV[1])
			self.V[1] += (dV1 + self.dV[1]) / (2.0 * self.sample_rate)
			self.dV[1] = dV1
			self.tV[1] = np.tanh(self.V[1] / (2.0 * VT))
			
			dV2 = self.g * (self.tV[1] - self.tV[2])
			self.V[2] += (dV2 + self.dV[2]) / (2.0 * self.sample_rate)
			self.dV[2] = dV2
			self.tV[2] = np.tanh(self.V[2] / (2.0 * VT))
			
			dV3 = self.g * (self.tV[2] - self.tV[3])
			self.V[3] += (dV3 + self.dV[3]) / (2.0 * self.sample_rate)
			self.dV[3] = dV3
			self.tV[3] = np.tanh(self.V[3] / (2.0 * VT))
			
			samples[i] = self.V[3]

		return samples
	
	def setCutoff(self, cutoff):
		self.cutoff = cutoff
		self.x = (MOOG_PI * cutoff) / self.sample_rate
		self.g = 4.0 * MOOG_PI * VT * cutoff * (1.0 - self.x) / (1.0 + self.x)
