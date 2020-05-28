import click
import librosa
import numpy as np
import soundfile as sf


@click.command()
@click.option('--file', required=True)
@click.option('--st', default=0, help='number of semitones to shift')
def pitch(file):
	# have options here with core.resample...
	#y, s = librosa.load('./test.mp3', sr=)


if __name__ == '__main__':
	pitch()
