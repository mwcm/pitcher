import sox
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


OUTPUT_FILE_NAME = 'aeiou.wav'
# INPUT_SAMPLE_RATE = 44100
NORMALIZED_DB = [-32, -18]
INPUT_SAMPLE_RATE = 96000
OUTPUT_SAMPLE_RATE = 48000

ZERO_ORDER_HOLD_MULTIPLIER = 4
RESAMPLE_MULTIPLIER = 2
TARGET_SAMPLE_RATE = 26040
TARGET_SAMPLE_RATE_MULTIPLE = TARGET_SAMPLE_RATE * RESAMPLE_MULTIPLIER

# map these if they grow any longer
PITCH_METHODS = ['manual', 'rubberband']
RESAMPLE_METHODS = ['librosa', 'scipy']


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# TODO: 12 bit
# https://en.wikipedia.org/wiki/Audio_bit_depth
# a way to simulate this could be to reduce the signal to noise ratio
# ie from 16 bits to 12 by scaling the volume by 0.0625 then normalize
# - avoid dithering in either operation

# http://mural.maynoothuniversity.ie/4115/1/40.pdf

# signal path: input filter > sample & hold > 12 bit quantizer > pitching > zero order
# hold > optional eq filters > output filter

# 4 total resamples:
# - input to 96khz
# - resample to multiple of sp 1200 rate
# - resample to sp 1200 rate
# - resample to 48khz for output


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
    resampled = librosa.core.resample(y, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE_MULTIPLE)
    decimated = sp.signal.decimate(resampled, RESAMPLE_MULTIPLIER)
    return decimated


def zero_order_hold(y):
    # zero order hold, TODO: test all this properly
    print(y)
    zero_hold_step1 = np.repeat(y, ZERO_ORDER_HOLD_MULTIPLIER)
    # or
    # zero_hold_step1 = np.fromiter((pitched[int(i)] for i in np.linspace(0, len(pitched)-1, num=len(pitched) * ZERO_ORDER_HOLD_MULTIPLIER)), np.float32)
    print(zero_hold_step1)

    # TODO Should we do a decimate step here? or combine with "resample for
    # output filter" step?
    zero_hold_step2 = sp.signal.decimate(zero_hold_step1,
                                         ZERO_ORDER_HOLD_MULTIPLIER)
    # or
    # zero_hold_step2 = librosa.core.resample(zero_hold_step1, TARGET_SAMPLE_RATE * ZERO_ORDER_HOLD_MULTIPLIER, TARGET_SAMPLE_RATE)
    print(zero_hold_step2)
    return zero_hold_step2


def bit_reduction(resampled):
    t = sox.Transformer()
    t.vol(0.0625)  # 4096 / 65,536 https://en.wikipedia.org/wiki/Audio_bit_depth
    t.norm()
    status, y_out, err = t.build(input_array=resampled, sample_rate_in=TARGET_SAMPLE_RATE)
    return y_out


# NOTE: maybe skip the anti aliasing?
# http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
# The sample input goes via an anti-aliasing filter to remove unwanted
# frequencies that are above half the sample frequency,
# the cutoff is brick walled at 42dB.


@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
@click.option('--pitch_method', default='manual_pitch')
@click.option('--resample_method', default='librosa')
def pitch(file, st, pitch_method, resample_method):

    # resample #1
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

    # change to 12 bit
    bit_reduced = bit_reduction(resampled)

    if pitch_method in PITCH_METHODS:
        if pitch_method == PITCH_METHODS[0]:
            pitched = manual_pitch(bit_reduced, st)
        elif pitch_method == PITCH_METHODS[1]:
            pitched = pyrb_pitch(bit_reduced, st)
    else:
        raise ValueError(f'invalid pitch method, valid methods are {PITCH_METHODS}')

    post_zero_order_hold = zero_order_hold(pitched)

    # TODO SSM-2044 here , find & adjust moog ladder filter code

    # resample for output filter
    # TODO investigate the exception that arises when fortranarray cast is rm'd
    output = librosa.core.resample(np.asfortranarray(post_zero_order_hold),
                                   TARGET_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

    # TODO: should have an output filter here, link above
    sf.write(OUTPUT_FILE_NAME, output, OUTPUT_SAMPLE_RATE,
             format='WAV', subtype='PCM_16')


if __name__ == '__main__':
    pitch()
