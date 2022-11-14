#! /usr/bin/env python3

import click
from pitcher import pitch, OUTPUT_FILTER_TYPES


@click.command()
@click.option('--st',                         type=int,     default=0, help='number of semitones to shift')
@click.option('--input-file',                 type=str,     required=True)
@click.option('--output-file',                type=str,     required=True)
@click.option('--log-level',                  type=str,     default='INFO')
@click.option('--input-filter',               is_flag=True, default=True)
@click.option('--quantize',                   is_flag=True, default=True)
@click.option('--time-stretch',               is_flag=True, default=True)
@click.option('--output-filter',              is_flag=True, default=True)
@click.option('--normalize-output',           is_flag=True, default=False)
@click.option('--quantize-bits',              type=int,     default=12, help='bit rate of quantized output')
@click.option('--custom-time-stretch',        type=float,   default=1.0)
@click.option('--output-filter-type',         type=click.Choice(OUTPUT_FILTER_TYPES), default=OUTPUT_FILTER_TYPES[0])
@click.option('--moog-output-filter-cutoff',  type=int,     default=10000)
@click.option('--force_mono',                 is_flag=True, default=False)
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
        force_mono
    ):
    pitch(
        st=st,
        input_file=input_file,
        output_file=output_file,
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
        force_mono=force_mono
    )
    return



if __name__ == '__main__':
   cli_wrapper() 
