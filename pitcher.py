#! /usr/bin/env python3
# Pitcher v 0.1
# Copyright (C) 2020 Morgan Mitchell
# Based on: Physical and Behavioral Circuit Modeling of the SP-12, DT Yeh, 2007
# https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf


import logging
import click
import numpy as np
import scipy as sp
import audiofile as af

from librosa.core import resample
from librosa import load

ZOH_MULTIPLIER = 4
RESAMPLE_MULTIPLIER = 2

INPUT_SR = 96000
OUTPUT_SR = 48000
TARGET_SR = 26040

POSITIVE_TUNING_RATIO = 1.02930223664
NEGATIVE_TUNING_RATIOS = {-1: 1.05652677103003,
                          -2: 1.1215356033380033,
                          -3: 1.1834835840896631,
                          -4: 1.253228360845465,
                          -5: 1.3310440397149297,
                          -6: 1.4039714929646099,
                          -7: 1.5028019735639886,
                          -8: 1.5766735700797954}

valid_log_levels = ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']


def calc_quantize_function(quantize_bits, log):
    # https://dspillustrations.com/pages/posts/misc/quantization-and-quantization-noise.html
    log.info(f'calculating quantize fn with {quantize_bits} quantize bits')
    u = 1  # max amplitude to quantize
    quantization_levels = 2 ** quantize_bits
    delta_s = 2 * u / quantization_levels  # level distance
    s_midrise = -u + delta_s / 2 + np.arange(quantization_levels) * delta_s
    s_midtread = -u + np.arange(quantization_levels) * delta_s
    log.info('done calculating quantize fn')
    return s_midrise, s_midtread


def adjust_pitch(x, st, log):
    log.info(f'adjusting audio pitch by {st} semitones')
    if (0 > st >= -8):
        t = NEGATIVE_TUNING_RATIOS[st]
    elif st > 0:
        t = POSITIVE_TUNING_RATIO ** -st
    elif st == 0:  # no change
        return x
    else:  # -8 > st: extrapolate, seems to lose a few points of precision?
        f = sp.interpolate.interp1d(list(NEGATIVE_TUNING_RATIOS.keys()),
                                    list(NEGATIVE_TUNING_RATIOS.values()),
                                    fill_value='extrapolate')
        t = f(st)

    n = int(np.round(len(x) * t))
    r = np.linspace(0, len(x) - 1, n).round().astype(np.int32)
    pitched = [x[r[e]] for e in range(n-1)]  # could yield here instead
    log.info('done pitching audio')
    return pitched


def filter_input(x, log):
    log.info('applying anti aliasing filter')
    # approximating the anti aliasing filter, don't think this needs to be
    # perfect since at fs/2=13.02kHz only -10dB attenuation, might be able to
    # improve accuracy in the 15 -> 20kHz range with firwin?
    f = sp.signal.ellip(4, 1, 72, 0.666, analog=False, output='sos')
    y = sp.signal.sosfilt(f, x)
    log.info('done applying anti aliasing filter')
    return y


# could use sosfiltfilt for zero phase filtering, but it doubles filter order
def filter_output(x, log):
    log.info('applying output eq filter')
    freq = np.array([0, 6510, 8000, 10000, 11111, 13020, 15000, 17500, 20000, 24000])
    att = np.array([0, 0, -5, -10, -15, -23, -28, -35, -41, -40])
    gain = np.power(10, att/20)
    f = sp.signal.firwin2(45, freq, gain, fs=OUTPUT_SR, antisymmetric=False)
    sos = sp.signal.tf2sos(f, [1.0])
    y = sp.signal.sosfilt(sos, x)
    log.info('done applying output eq filter')
    return y


def scipy_resample(y, input_sr, target_sr, factor, log):
    ''' resample from input_sr to target_sr_multiple/factor'''
    log.info(f'resampling audio to sample rate of {target_sr * factor}')
    seconds = len(y)/input_sr
    target_samples = int(seconds * (target_sr * factor)) + 1
    resampled = sp.signal.resample(y, target_samples)
    log.info('done resample 1/2')
    log.info(f'resampling audio to sample rate of {target_sr}')
    decimated = sp.signal.decimate(resampled, factor)
    log.info('done resample 2/2')
    log.info('done resampling audio')
    return decimated


def zero_order_hold(y, zoh_multiplier, log):
    log.info(f'applying zero order hold of {zoh_multiplier}')
    # intentionally oversample by repeating each sample 4 times
    # could also try a freq aliased sinc filter
    zoh_applied = np.repeat(y, zoh_multiplier)
    log.info('done applying zero order hold')
    return zoh_applied


def nearest_values(x, y):
    x, y = map(np.asarray, (x, y))
    tree = sp.spatial.cKDTree(y[:, None])
    ordered_neighbors = tree.query(x[:, None], 1)[1]
    return ordered_neighbors


# no audible difference after audacity invert test @ 12 bits
# however, when plotted the scaled amplitude of quantized audio is
# noticeably higher than the original, leaving for now
def quantize(x, S, bits, log):
    log.info('quantizing audio @ {bits} bits')
    y = nearest_values(x, S)
    quantized = S.flat[y].reshape(x.shape)
    log.info('done quantizing')
    return quantized


@click.command()
@click.option('--st', default=0, help='number of semitones to shift')
@click.option('--log-level', default='INFO')
@click.option('--input-file', required=True)
@click.option('--output-file', required=True)
@click.option('--quantize-bits', default=12, help='bit rate of quantized output')
@click.option('--skip-quantize', is_flag=True, default=False)
@click.option('--skip-normalize', is_flag=True, default=False)
@click.option('--skip-input-filter', is_flag=True, default=False)
@click.option('--skip-output-filter', is_flag=True, default=False)
def pitch(st, log_level, input_file, output_file, quantize_bits, skip_normalize,
          skip_quantize, skip_input_filter, skip_output_filter):

    log = logging.getLogger(__name__)

    if (not log_level) or (log_level.upper() not in valid_log_levels):
        log_level = 'INFO'
        log.warn(f'Invalid log-level: "{log_level}", log-level set to "INFO", '
                 f'valid log levels are {valid_log_levels}')

    log.setLevel(log_level)

    log.info(f'loading: "{input_file}" at oversampled rate: {INPUT_SR}')
    y, s = load(input_file, sr=INPUT_SR)
    log.info('done loading')

    midrise, midtread = calc_quantize_function(quantize_bits, log)

    if skip_input_filter:
        log.info('skipping input anti aliasing filter')
    else:
        y = filter_input(y)

    resampled = scipy_resample(y, INPUT_SR, TARGET_SR, RESAMPLE_MULTIPLIER, log)

    if skip_quantize:
        log.info('skipping quantize')
    else:
        # simulate analog -> digital conversion
        # TODO: midtread/midrise option?
        resampled = quantize(resampled, midtread, quantize_bits, log)

    pitched = adjust_pitch(resampled, st, log)

    # oversample again (default factor of 4) to simulate ZOH
    # TODO: retest output against freq aliased sinc fn
    post_zero_order_hold = zero_order_hold(pitched, log)

    # TODO: try using scipy resample here?
    output = resample(np.asfortranarray(post_zero_order_hold),
                      TARGET_SR * ZOH_MULTIPLIER, OUTPUT_SR)

    if skip_output_filter:
        log.info('skipping output eq filter')
    else:
        output = filter_output(output, log)  # eq filter

    log.info(f'writing {output_file}, at sample rate: {OUTPUT_SR} '
             f'with skip_normalize set to: {skip_normalize}')

    af.write(output_file, output, OUTPUT_SR, '16bit', not skip_normalize)

    log.info(f'done! output_file at: {output_file}')
    return


if __name__ == '__main__':
    pitch()
