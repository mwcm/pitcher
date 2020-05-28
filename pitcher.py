import click
import librosa
import numpy as np
import soundfile as sf

ST_POSITIVE = 1.02930223664
ST_NEGATIVE = {-1: 1.05652677103003,
               -2: 1.1215356033380033,
               -3: 1.1834835840896631,
               -4: 1.253228360845465,
               -5: 1.3310440397149297,
               -6: 1.4039714929646099,
               -7: 1.5028019735639886,
               -8: 1.5766735700797954}

INPUT_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 26040

# TODO
# http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
# The sample input goes via an anti-aliasing filter to remove unwanted frequencies that are above half the sample frequency, the cutoff is brick walled at 42dB.


# TODO
# https://ccrma.stanford.edu/~dtyeh/papers/yeh07_icmc_sp12.pdf
# To simulate aliasing accurately using a digital implemen-tation,  the  discrete-time  signal  is  ideally  interpolated  tothe  time-grid  corresponding  to  the  sampling  rate  of  theSP-12.Methods to approximate the ideal interpolation includeusing a variable delay filter [2], or resampling[5] to a mul-tiple of the SP-12’s sampling rate and then downsamplingto the SP-12 rate.


# TODO: pitching up works, but 0 st outputs lower pitched file
# TODO: librosa resamples on load, what was the JS behaviour?



def manual_pitch(y):

def auto_pitch(y)
    pitched = librosa.effects.pitch_shift(y, TARGET_SAMPLE_RATE, n_steps=st)
    return pitched

def time_shift(y):
    pitched = librosa.effects.time_stretch(y, TARGET_SAMPLE_RATE, n_steps=st)
    return pitched


@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
def pitch(file, st):
    # if (0 > st >= -8):
    #     t = ST_NEGATIVE[st]
    # elif (st >= 0):
    #     t = ST_POSITIVE ** -st
    # else:
    #     raise Exception('invalid semitone count')

    y, s = librosa.load(file, sr=INPUT_SAMPLE_RATE)

    y = librosa.core.resample(y, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE)

    # n = int(np.round(len(y) * t))
    # r = np.linspace(0, len(y), n)
    # new = np.zeros(n, dtype=np.float32)
    #
    # for e in range(int(n) - 1):
    #     new[e] = y[int(np.round(r[e]))]

    sf.write('./aeiou.wav', new, TARGET_SAMPLE_RATE, format='wav')


if __name__ == '__main__':
    pitch()
