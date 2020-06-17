import click
import numpy as np
import scipy as sp
import audiofile as af

from librosa.core import resample
from librosa import load

POSITIVE_TUNING_RATIO = 1.02930223664
NEGATIVE_TUNING_RATIOS = {-1: 1.05652677103003,
                          -2: 1.1215356033380033,
                          -3: 1.1834835840896631,
                          -4: 1.253228360845465,
                          -5: 1.3310440397149297,
                          -6: 1.4039714929646099,
                          -7: 1.5028019735639886,
                          -8: 1.5766735700797954}

# https://dspillustrations.com/pages/posts/misc/quantization-and-quantization-noise.html
U = 1  # max amplitude to quantize
QUANTIZATION_BITS = 12
QUANTIZATION_LEVELS = 2 ** QUANTIZATION_BITS
DELTA_S = 2 * U / QUANTIZATION_LEVELS  # level distance
S_MIDRISE = -U + DELTA_S / 2 + np.arange(QUANTIZATION_LEVELS) * DELTA_S
S_MIDTREAD = -U + np.arange(QUANTIZATION_LEVELS) * DELTA_S

RESAMPLE_MULTIPLIER = 2
ZOH_MULTIPLIER = 4

INPUT_SR = 96000
OUTPUT_SR = 48000
TARGET_SR = 26040
TARGET_SR_MULTIPLE = TARGET_SR * RESAMPLE_MULTIPLIER

RESAMPLE_METHODS = ['librosa', 'scipy']


def adjust_pitch(x, st):
    if (0 > st >= -8):
        t = NEGATIVE_TUNING_RATIOS[st]
    elif st > 0:
        t = POSITIVE_TUNING_RATIO ** -st
    elif st == 0:  # no change
        return x
    else:  # -8 > st: extrapolate, seems to lose a few points of percision?
        f = sp.interpolate.interp1d(list(NEGATIVE_TUNING_RATIOS.keys()),
                                    list(NEGATIVE_TUNING_RATIOS.values()),
                                    fill_value='extrapolate')
        t = f(st)

    n = int(np.round(len(x) * t))
    r = np.linspace(0, len(x) - 1, n).round().astype(np.int32)
    return [x[r[e]] for e in range(n-1)]  # could yield here


def filter_input(x):
    # approximating the anti aliasing filter, don't think this needs to be
    # perfect since at fs/2=13.02kHz only -10dB attenuation, might be able to
    # improve accuracy in the 15 -> 20kHz range with firwin?
    f = sp.signal.ellip(4, 1, 72, 0.666, analog=False, output='sos')
    y = sp.signal.sosfilt(f, x)
    return y


def filter_output(x):
    freq = np.array([0, 6510, 8000, 10000, 11111, 13020, 15000, 17500, 20000, 24000])
    att = np.array([0, 0, -5, -10, -15, -23, -28, -35, -41, -40])
    gain = np.power(10, att/20)
    f = sp.signal.firwin2(45, freq, gain, fs=OUTPUT_SR, antisymmetric=False)
    sos = sp.signal.tf2sos(f, [1.0])
    y = sp.signal.sosfilt(sos, x)
    return y


def librosa_resample(y):
    resampled = resample(y, INPUT_SR, TARGET_SR_MULTIPLE)
    downsampled = resample(resampled, TARGET_SR_MULTIPLE, TARGET_SR)
    return downsampled


def scipy_resample(y):
    seconds = len(y)/INPUT_SR
    target_samples = int(seconds * TARGET_SR_MULTIPLE) + 1
    resampled = sp.signal.resample(y, target_samples)
    decimated = sp.signal.decimate(resampled, RESAMPLE_MULTIPLIER)
    return decimated


def zero_order_hold(y):
    # intentionally oversample by repeating each sample 4 times
    # could also try a freq aliased sinc filter
    return np.repeat(y, ZOH_MULTIPLIER)


def nearest_values(x, y):
    x, y = map(np.asarray, (x, y))
    tree = sp.spatial.cKDTree(y[:, None])
    ordered_neighbors = tree.query(x[:, None], 1)[1]
    return ordered_neighbors


# no audible difference after audacity invert test @ 12 bits
# however, when plotted the scaled amplitude of quantized audio is
# noticeably higher than the original
def quantize(x, S):
    y = nearest_values(x, S)
    quantized = S.flat[y].reshape(x.shape)
    return quantized


# same issue as SAR, output needs to be rescaled
# we'd like output to be the same scale as input, just quantized
def digitize(x, S):
    y = np.digitize(x.flatten(), S.flatten())
    return y


# TODO
# - logging
# - add cli option for quantization bits
# - optionally preserve stereo channels throughout processing
# - impletement optional vcf? (ring moog) good description in slides
# - improve input anti aliasing filter fit
# - replace or delete pyrb
# - replace librosa if there is a module with better performance, maybe essentia?
# - supress librosa numba warning

# NOTES:
# - could use sosfiltfilt for zero phase filtering, but it doubles filter order

# Based one
# https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf

# signal path: input filter > sample & hold > 12 bit quantizer > pitching
# & decay > zero order hold > optional eq filters > output filter

@click.command()
@click.option('--st', default=0, help='number of semitones to shift')
@click.option('--normalize', is_flag=True, default=False)
@click.option('--input-file', required=True)
@click.option('--output-file', required=True)
@click.option('--resample-fn', default='scipy')
@click.option('--skip-quantize', is_flag=True, default=False)
@click.option('--skip-input-filter', is_flag=True, default=False)
@click.option('--skip-output-filter', is_flag=True, default=False)
def pitch(st, normalize, input_file, output_file, resample_fn,
          skip_quantize, skip_input_filter, skip_output_filter):

    y, s = load(input_file, sr=INPUT_SR)

    if not skip_input_filter:
        y = filter_input(y)

    # TODO: should indicate sample rates here rather than bury in function
    if resample_fn in RESAMPLE_METHODS:
        if resample_fn == RESAMPLE_METHODS[0]:
            resampled = librosa_resample(y)
        elif resample_fn == RESAMPLE_METHODS[1]:
            resampled = scipy_resample(y)
    else:
        raise ValueError('invalid resample method, '
                         f'valid methods are {RESAMPLE_METHODS}')

    if not skip_quantize:
        # simulate analog -> digital conversion
        resampled = quantize(resampled, S_MIDRISE)  # TODO: midtread/midrise?

    pitched = adjust_pitch(resampled, st)

    # oversample again (default factor of 4) to simulate ZOH
    # TODO: retest output, test freq aliased sinc fn
    post_zero_order_hold = zero_order_hold(pitched)

    # give option use scipy resample here?
    output = resample(np.asfortranarray(post_zero_order_hold),
                      TARGET_SR * ZOH_MULTIPLIER, OUTPUT_SR)

    if not skip_output_filter:
        output = filter_output(output)  # equalization filter

    af.write(output_file, output, OUTPUT_SR, '16bit', normalize)


if __name__ == '__main__':
    pitch()
