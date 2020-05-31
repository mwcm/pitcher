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

# TODO quantization bits should be an option
QUANTIZATION_BITS = 8
QUANTIZATION_LEVELS = 2 ** QUANTIZATION_BITS
U = 1  # max Amplitude to be quantized TODO: Revisit
DELTA_S = 2 * U / QUANTIZATION_LEVELS  # level distance

S_MIDRISE = -U + DELTA_S / 2 + np.arange(QUANTIZATION_LEVELS) * DELTA_S
S_MIDTREAD = -U + np.arange(QUANTIZATION_LEVELS) * DELTA_S

INPUT_SAMPLE_RATE = 96000
OUTPUT_SAMPLE_RATE = 48000

ZERO_ORDER_HOLD_MULTIPLIER = 4
RESAMPLE_MULTIPLIER = 1
TARGET_SAMPLE_RATE = 26040
TARGET_SAMPLE_RATE_MULTIPLE = TARGET_SAMPLE_RATE * RESAMPLE_MULTIPLIER

# map these if they grow any longer
PITCH_METHODS = ['manual', 'rubberband']
RESAMPLE_METHODS = ['librosa', 'scipy']

# https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf
# http://mural.maynoothuniversity.ie/4115/1/40.pdf

# signal path: input filter > sample & hold > 12 bit quantizer > pitching
# & decay > zero order hold > optional eq filters > output filter

# 4 total resamples:
# - input to 96khz
# - resample to multiple of sp 1200 rate
# - resample to sp 1200 rate
# - resample to 48khz for output


def sizeof_fmt(num, suffix='B'):
    # https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def manual_pitch(y, st):
    if (0 > st >= -8):
        t = ST_NEGATIVE[st]
    elif (st >= 0):
        t = ST_POSITIVE ** -st
    else:
        t = ST_POSITIVE ** (-st + 8)  # TODO: rough guess, revisit

    n = int(np.round(len(y) * t))
    r = np.linspace(0, len(y), n)
    new = np.zeros(n, dtype=np.float32)

    for e in range(n - 1):
        new[e] = y[int(np.round(r[e]))]

    return new


def pyrb_pitch(y, st):
    t = ST_POSITIVE ** st  # close enough to og vals? maybe revisit
    pitched = pyrb.pitch_shift(y, TARGET_SAMPLE_RATE, n_steps=st)
    return librosa.effects.time_stretch(pitched, t)


def librosa_resample(y):
    # http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
    # "...resample to a multiple of the SP-12(00)'s sampling rate..."
    resampled = librosa.core.resample(y, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE_MULTIPLE)
    # "...then downsample to the SP-12(00) rate"
    downsampled = librosa.core.resample(resampled, TARGET_SAMPLE_RATE_MULTIPLE, TARGET_SAMPLE_RATE)
    return downsampled


def scipy_resample(y):
    seconds = len(y)/INPUT_SAMPLE_RATE
    target_samples = int(seconds * TARGET_SAMPLE_RATE) + 1
    resampled = sp.signal.resample(y, target_samples)
    decimated = sp.signal.decimate(resampled, RESAMPLE_MULTIPLIER)
    return decimated


def zero_order_hold(y):
    # zero order hold, TODO: come back & test all this properly, see sp-12 slides
    zero_hold_step1 = np.repeat(y, ZERO_ORDER_HOLD_MULTIPLIER)
    # or
    # zero_hold_step1 = np.fromiter((pitched[int(i)] for i in np.linspace(0, len(pitched)-1, num=len(pitched) * ZERO_ORDER_HOLD_MULTIPLIER)), np.float32)

    # TODO Decimate step here? or combine with "resample for output filter" step?
    #      Or no decimate at all? In that case how do we get the post ZOH to a good length?
    zero_hold_step2 = sp.signal.decimate(zero_hold_step1,
                                         ZERO_ORDER_HOLD_MULTIPLIER)
    # or
    # zero_hold_step2 = librosa.core.resample(zero_hold_step1, TARGET_SAMPLE_RATE * ZERO_ORDER_HOLD_MULTIPLIER, TARGET_SAMPLE_RATE)
    return zero_hold_step2


# TODO: make optional
# TODO: MemoryError: Unable to allocate... on abs(X-S)
# TODO: why does array diff take so much ram at high quantize bits?
#       the size of the arrays are large, but this can't be that hard to do
#       efficiently, needs optimization
# NOTE: not a huge effect on the sound above 8bits, fun to play around with tho
def quantize(x, S):

    # make things a little quicker _for now_ by using float16
    # there are better ways to optimize this
    x = x.astype(np.float16)
    S = S.astype(np.float16)

    X = x.reshape((-1, 1))
    # S = S.reshape((1, -1)) # don't think this is necessary?

    # TODO:
    # 1. how to do these in a way that results in nearestIndex &
    #    avoids storing the entire array of distributions in memory at once?
    # 2. how to do the operations most efficiently?
    dists = abs(X-S)
    nearestIndex = dists.argmin(axis=1)

    quantized = S.flat[nearestIndex]
    quantized = quantized.reshape(x.shape)
    print(sizeof_fmt(quantized.nbytes))
    return quantized


# NOTE: maybe skip the anti aliasing?
# http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
# The sample input goes via an anti-aliasing filter to remove unwanted
# frequencies that are above half the sample frequency,
# the cutoff is brick walled at 42dB.
@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
@click.option('--pitch_method', default='manual')
@click.option('--resample_method', default='scipy')
@click.option('--output_file', required=True)
def pitch(file, st, pitch_method, resample_method, output_file):

    # resample #1, purposefully oversample to 96khz
    y, s = librosa.load(file, sr=INPUT_SAMPLE_RATE)

    # TODO: input anti alias filter here fig 2 in sp-12 paper or sp-1200's above
    # https://dsp.stackexchange.com/questions/2864/how-to-write-lowpass-filter-for-sampled-signal-in-python
    # then anti alias w/ order 11

    # resample #2 & #3
    if resample_method in RESAMPLE_METHODS:
        if resample_method == RESAMPLE_METHODS[0]:
            resampled = librosa_resample(y)
        elif resample_method == RESAMPLE_METHODS[1]:
            resampled = scipy_resample(y)
    else:
        raise ValueError(f'invalid resample method, valid methods are {RESAMPLE_METHODS}')

    # simulate 12 bit adc conversion
    bit_reduced = quantize(resampled, S_MIDTREAD)  # TODO: midtread or midrise?

    if pitch_method in PITCH_METHODS:
        if pitch_method == PITCH_METHODS[0]:
            pitched = manual_pitch(bit_reduced, st)
        elif pitch_method == PITCH_METHODS[1]:
            pitched = pyrb_pitch(bit_reduced, st)
    else:
        raise ValueError(f'invalid pitch method, valid methods are {PITCH_METHODS}')

    post_zero_order_hold = zero_order_hold(pitched)

    # TODO optional SSM-2044 here, find & adjust moog ladder filter code

    # resample for output filter
    # TODO investigate the exception that arises when fortranarray cast is rm'd
    output = librosa.core.resample(np.asfortranarray(post_zero_order_hold),
                                   TARGET_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

    # TODO: should have an output filter here, link above
    sf.write(output_file, output, OUTPUT_SAMPLE_RATE,
             format='WAV', subtype='PCM_16')


if __name__ == '__main__':
    pitch()
