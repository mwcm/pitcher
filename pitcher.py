import click
import librosa
import numpy as np
import scipy as sp
import soundfile as sf

from numba import jit
from pyrubberband import pyrb

ST_POSITIVE = 1.02930223664
# TODO: revisit, "sp-12 has 32 different tuning settings with various skip amounts"
# skip of ~0.64062 creates nice aliasing
ST_NEGATIVE = {-1: 1.05652677103003,
               -2: 1.1215356033380033,
               -3: 1.1834835840896631,
               -4: 1.253228360845465,
               -5: 1.3310440397149297,
               -6: 1.4039714929646099,
               -7: 1.5028019735639886,
               -8: 1.5766735700797954}

QUANTIZATION_BITS = 12
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


def manual_pitch(x, st):
    if (0 > st >= -8):
        t = ST_NEGATIVE[st]
    elif (st >= 0):
        t = ST_POSITIVE ** -st
    else:
        t = ST_POSITIVE ** (-st + 8)  # TODO: rough guess, revisit

    n = int(np.round(len(x) * t))
    r = np.linspace(0, len(x), n)
    new = np.zeros(n, dtype=np.float32)

    for e in range(n - 1):
        new[e] = x[int(np.round(r[e]))]

    return new


def filter_input(x):
    # approximating the anti aliasing filter, don't think this needs to be
    # perfect since at fs/2=13.02kHz only -10dB attenuation, might be able to
    # improve accuracy with firwin
    f = sp.signal.ellip(4, 1, 72, 0.666, analog=False, output='sos')
    y = sp.signal.sosfilt(f, x)
    return y


# https://patrickignoto.com/2017/04/11/mumt-618-final-project/
# for i = 1:N
#     # Input to first LP filter stage (after waveshaping)
#     input = tanh(x(i) - 4*Gres*(ym1(4) - Gcomp * x(i)));
#     # Send input through 4 stages of LP filtering
#     for n = 1:4
#         # Output of each stage
#         output = ((glp1*input + glp2*xm1(n) - ym1(n))*g) + ym1(n);
#         xm1(n) = input;     # store x[n-1] for each stage
#         ym1(n) = output;    #store y[n-1] for each stage
#         input = output;     #input to next stage is this stage's output
#     end
#
#     y(i) = output;
# end

def filter_output(x):
    # use window method to replicate the fixed output filter
    freq = np.array([0, 6510, 8000, 10000, 11111, 13020, 15000, 17500, 20000, 24000])
    att = np.array([0, 0, -5, -10, -15, -23, -28, -35, -40, -40])
    gain = np.power(10, att/20)
    f = sp.signal.firwin2(45, freq, gain, fs=OUTPUT_SAMPLE_RATE, antisymmetric=False)
    sos = sp.signal.tf2sos(f, [1.0])
    y = sp.signal.sosfilt(sos, x)  # can use sosfiltfilt if phase is an issue but will result in 2 * filter order
    return y


def pyrb_pitch(y, st):
    t = ST_POSITIVE ** st  # close enough to og vals? maybe revisit
    pitched = pyrb.pitch_shift(y, TARGET_SAMPLE_RATE, n_steps=st)
    return librosa.effects.time_stretch(pitched, t)


def librosa_resample(y):
    # http://www.synthark.org/Archive/EmulatorArchive/SP1200.html
    # "...resample to a multiple of the SP-12(00)'s sampling rate..."
    resampled = librosa.core.resample(y, INPUT_SAMPLE_RATE,
                                      TARGET_SAMPLE_RATE_MULTIPLE)
    # "...then downsample to the SP-12(00) rate"
    downsampled = librosa.core.resample(resampled, TARGET_SAMPLE_RATE_MULTIPLE,
                                        TARGET_SAMPLE_RATE)
    return downsampled


def scipy_resample(y):
    seconds = len(y)/INPUT_SAMPLE_RATE
    target_samples = int(seconds * TARGET_SAMPLE_RATE) + 1
    resampled = sp.signal.resample(y, target_samples)

    decimated = sp.signal.decimate(resampled, RESAMPLE_MULTIPLIER)
    return decimated


# TODO: come back & test all this properly, see sp-12 slides
def zero_order_hold(y):
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


# NOTE: not much effect on the sound above 12bits
def quantize(x, S):

    X = x.reshape((-1, 1))
    S = S.reshape((1, -1))  # don't think this is necessary

    @jit(nopython=True)
    def compute_distributions(X, S):
        y = np.zeros(len(X), dtype=np.int64)  # TODO: remove int64
        for i, item in enumerate(X):
            dists = np.abs(item-S)
            nearestIndex = np.argmin(dists)
            y[i] = nearestIndex
        return y

    y = compute_distributions(X, S)
    quantized = S.flat[y]
    return quantized.reshape(x.shape)


# TODO
# - redo output filter
# - try adding ring mod ?
# - revisit zoh
# - better logging
# - requirements
# - readme
# - revisit pitch values
# - re-test chunking performance on full songs
# - replace pyrb
# - supress numba warning

# Based on:
# https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf

# signal path: input filter > sample & hold > 12 bit quantizer > pitching
# & decay > zero order hold > optional eq filters > output filter

# 4 total resamples:
# - input to 96khz
# - resample to multiple of sp 1200 rate
# - resample to sp 1200 rate
# - resample to 48khz for output
@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
@click.option('--pitch-method', default='manual')
@click.option('--resample-method', default='scipy')
@click.option('--output-file', required=True)
@click.option('--skip-input-filter', is_flag=True, default=False)
@click.option('--skip-output-filter', is_flag=True, default=False)
@click.option('--skip-quantize', is_flag=True, default=False)
@click.option('--skip-normalize', is_flag=True, default=False)
def pitch(file, st, pitch_method, resample_method, output_file,
          skip_input_filter, skip_output_filter, skip_quantize,
          skip_normalize):

    # resample #1, purposefully oversample to 96khz
    y, s = librosa.load(file, sr=INPUT_SAMPLE_RATE)

    if not skip_input_filter:
        y = filter_input(y)

    # resample #2 & #3
    if resample_method in RESAMPLE_METHODS:
        if resample_method == RESAMPLE_METHODS[0]:
            resampled = librosa_resample(y)
        elif resample_method == RESAMPLE_METHODS[1]:
            resampled = scipy_resample(y)
    else:
        raise ValueError('invalid resample method, '
                         f'valid methods are {RESAMPLE_METHODS}')

    # simulate 12 bit adc conversion
    if not skip_quantize:
        bit_reduced = quantize(resampled, S_MIDTREAD)  # TODO: midtread or midrise?

    if pitch_method in PITCH_METHODS:
        if pitch_method == PITCH_METHODS[0]:
            pitched = manual_pitch(bit_reduced, st)
        elif pitch_method == PITCH_METHODS[1]:
            pitched = pyrb_pitch(bit_reduced, st)
    else:
        raise ValueError('invalid pitch method, '
                         f'valid methods are {PITCH_METHODS}')

    post_zero_order_hold = zero_order_hold(pitched)

    # resample for output filter
    # TODO investigate the exception that arises when fortranarray cast is rm'd
    output = librosa.core.resample(np.asfortranarray(post_zero_order_hold),
                                   TARGET_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

    if not skip_output_filter:
        output = filter_output(output)

    if not skip_normalize:
        output = librosa.util.normalize(output)

    sf.write(output_file, output, OUTPUT_SAMPLE_RATE,
             format='WAV', subtype='PCM_16')


if __name__ == '__main__':
    pitch()
