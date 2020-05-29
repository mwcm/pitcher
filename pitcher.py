import click
import librosa
import numpy as np
import scipy as sp
import soundfile as sf


from pyrubberband import pyrb


ST_POSITIVE = 1.02930223664
ST_NEGATIVE = {-1: 1.05652677103003,
               -2: 1.1215356033380033,
               -3: 1.1834835840896631,
               -4: 1.253228360845465,
               -5: 1.3310440397149297,
               -6: 1.4039714929646099,
               -7: 1.5028019735639886,
               -8: 1.5766735700797954}

# INPUT_SAMPLE_RATE = 44100
NORMALIZED_DB = [-32, -18]
INPUT_SAMPLE_RATE = 96000

RESAMPLE_FACTOR = 2

TARGET_SAMPLE_RATE = 26040
TARGET_SAMPLE_RATE_MULTIPLE = TARGET_SAMPLE_RATE * RESAMPLE_FACTOR

PITCH_METHODS = ['manual', 'rubberband']
RESAMPLE_METHODS = ['librosa', 'scipy']

# TODO
# http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
# The sample input goes via an anti-aliasing filter to remove unwanted frequencies that are above half the sample frequency, the cutoff is brick walled at 42dB.


# TODO: librosa resamples on load, what was the original order
#       of resampling/pitching?


# TODO: allow for lower than -8 st
def manual_pitch(y, st):

    if (0 > st >= -8):
        t = ST_NEGATIVE[st]
    elif (st >= 0):
        t = ST_POSITIVE ** -st
    else:
        raise ValueError('invalid semitone count, should be 0 > st > -8')

    n = int(np.round(len(y) * t))
    r = np.linspace(0, len(y), n)
    new = np.zeros(n, dtype=np.float32)

    for e in range(n - 1):
        new[e] = y[int(np.round(r[e]))]

    return new


def pyrb_pitch(y, st):
    t = ST_POSITIVE ** st
    pitched = pyrb.pitch_shift(y, TARGET_SAMPLE_RATE, n_steps=st)
    return librosa.effects.time_stretch(pitched, t)


def librosa_resample(y):
    # http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
    # "...resample to a multiple of the SP-12(00)'s sampling rate..."
    y = librosa.core.resample(y, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE_MULTIPLE)

    # "...then downsample to the SP-12(00) rate"
    y = librosa.core.resample(y, TARGET_SAMPLE_RATE_MULTIPLE, TARGET_SAMPLE_RATE)
    return y


def scipy_resample(y):
     #y = sp.signal.decimate(y)
    return


@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
@click.option('--pitch_method', default='manual_pitch')
@click.option('--resample_method', default='librosa')
def pitch(file, st, method, resample_method):

    y, s = librosa.load(file, sr=INPUT_SAMPLE_RATE)

    if resample_method in RESAMPLE_METHODS:
        if method == RESAMPLE_METHODS[0]:
            new = librosa_resample(y)
        elif method == RESAMPLE_METHODS[1]:
            new = scipy_resample(y)
    else:
        raise ValueError(f'invalid resample method, valid methods are {RESAMPLE_METHODS}')

    if method in PITCH_METHODS:
        if method == PITCH_METHODS[0]:
            new = manual_pitch(y, st)
        elif method == PITCH_METHODS[1]:
            new = pyrb_pitch(y, st)
    else:
        raise ValueError(f'invalid pitch method, valid methods are {PITCH_METHODS}')

    # sf._subtypes['PCM_12'] = 0x0008
    sf.write('./aeiou.wav', new, TARGET_SAMPLE_RATE, format='WAV')


if __name__ == '__main__':
    pitch()
