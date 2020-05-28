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

# TODO: negative semitone values not working
# TODO: librosa resamples on load, what was the JS behaviour?


@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
def pitch(file, st):
    if (0 > st > -8):
        t = ST_NEGATIVE[st]
    elif (st > 0):
        t = ST_POSITIVE ** -st
    else:
        raise Exception('invalid semitone count')

    y, s = librosa.load(file, sr=INPUT_SAMPLE_RATE)

    n = np.round(len(y) * t)
    r = np.linspace(0, len(y), int(n))

    for e in range(0, int(n) - 1):
        y[int(e)] = y[np.round(int(r[int(e)]))]

    sf.write('./aeiou.ogg', y, TARGET_SAMPLE_RATE, format='ogg')


if __name__ == '__main__':
    pitch()
