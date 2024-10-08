from core import pitch, INPUT_SR

import click
from librosa import load
from pathlib import Path

# - could move this into core
# - would need to change core.py's st to an array
# - would also need to accomodate output_path - core expects a file path, not dir

def output_many(input_file, output_dir, up):
    in_file_name = Path(input_file).stem
    in_file_type = Path(input_file).suffix
    
    output_path = Path(output_dir)

    # pitch down 12 times, one per st drop
    OUTPUT_MANY_ST_RANGE = [x for x in range(-12, 0)]

    if up:
        # alternatively pitch up
        OUTPUT_MANY_ST_RANGE = [x for x in range(1, 12)]        

    
    if not output_path.exists():
        output_path.mkdir()
    
    if not output_path.is_dir():
        raise ValueError(f'output-dir should be a directory, received: {output_dir}') 

    input_file, s = load(input_file, sr=INPUT_SR, mono=False)

    for st in OUTPUT_MANY_ST_RANGE:
       output_file = f'{in_file_name}_{st}{in_file_type}'
       pitch(st=st, input_file_path='unused', output_file_path=str(output_path.joinpath(output_file)), log_level='INFO', input_data=input_file)

    return


@click.command()
@click.option('--input-file', type=str, required=True)
@click.option('--output-dir', type=str, required=True)
@click.option('--up', is_flag=True, default=False)
def wrapper(input_file, output_dir, up):
    output_many(input_file, output_dir, up)
    return


if __name__ == "__main__":
    wrapper()