#! /usr/bin/env python3

import click
from pitcher.core import pitch, OUTPUT_FILTER_TYPES

# NOTE: moog_output_filter_cutoff default could be adjusted to more closely match the low cutoff of the original
#       if that would be more useful
@click.command()
@click.option('--st',                         type=int,     default=0, help='number of semitones to shift pitch by')
@click.option('--input-file',                 type=str,     required=True, help='path to input audio file (WAV, MP3, OGG, FLAC)')
@click.option('--output-file',                type=str,     required=True, help='path to output audio file (format determined by extension)')
@click.option('--log-level',                  type=str,     default='INFO', help='logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
@click.option('--input-filter/--no-input-filter', default=True, help='apply/skip input anti-aliasing filter (default: --input-filter)')
@click.option('--quantize/--no-quantize',     default=True, help='apply/skip ADC quantization simulation (default: --quantize)')
@click.option('--time-stretch/--no-time-stretch', default=True, help='enable/disable time stretching (default: --time-stretch)')
@click.option('--output-filter/--no-output-filter', default=True, help='apply/skip output EQ filtering (default: --output-filter)')
@click.option('--normalize-output',           is_flag=True, default=False, help='normalize output audio')
@click.option('--quantize-bits',              type=int,     default=12, help='bit depth for quantization simulation (default: 12)')
@click.option('--custom-time-stretch',        type=float,   default=1.0, help='custom time stretch factor (1.0=device default, 0.0=none)')
@click.option('--output-filter-type',         type=click.Choice(OUTPUT_FILTER_TYPES), default=OUTPUT_FILTER_TYPES[0], help='output filter type: lp1 (7.5kHz cutoff), lp2 (10kHz cutoff), moog (SSM2044)')
@click.option('--moog-output-filter-cutoff',  type=int,     default=10000, help='cutoff frequency for moog (ssm2044) filter in Hz (20-20000, default 10000)')
@click.option('--force-mono',                 is_flag=True, default=False, help='convert input to mono (output will also be mono)')
@click.option('--use-sp12-rate',              is_flag=True, default=False, help='use SP-12 sample rate (27500 Hz) instead of SP-1200 (26040 Hz)')
def cli_wrapper(
        st,
        input_file,
        output_file,
        log_level,
        input_filter,
        quantize,
        time_stretch,
        output_filter,
        normalize_output,
        quantize_bits,
        custom_time_stretch,
        output_filter_type,
        moog_output_filter_cutoff,
        force_mono,
        use_sp12_rate
    ):

    pitch(
        st=st,
        input_file_path=input_file,
        output_file_path=output_file,
        log_level=log_level,
        input_filter=input_filter,
        quantize=quantize,
        time_stretch=time_stretch,
        output_filter=output_filter,
        normalize_output=normalize_output,
        quantize_bits=quantize_bits,
        custom_time_stretch=custom_time_stretch,
        output_filter_type=output_filter_type,
        moog_output_filter_cutoff=moog_output_filter_cutoff,
        force_mono=force_mono,
        use_sp12_rate=use_sp12_rate
    )
    return



if __name__ == '__main__':
   cli_wrapper() 
