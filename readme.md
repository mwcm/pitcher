# Pitcher.py

<img width="712" alt="Screen Shot 2022-11-14 at 8 09 32 PM" src="https://user-images.githubusercontent.com/2433319/201812501-af784d53-5a6d-4c94-af5d-1ffb2fc8cb11.png">


- Free & OS emulation of the SP-12 & SP-1200 signal chain (now with GUI)
- Pitch shift / bitcrush / resample audio files
- Written and tested in Python v3.10.7 on Windows 10 & MacOS Mojave 10.14.6
- Based on [Physical and Behavioral Circuit Modeling of the SP-12
Sampler, DT Yeh, 2007](https://ccrma.stanford.edu/~dtyeh/papers/yeh07_icmc_sp12.pdf) & [Slides](https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf)
- Audio examples [here](https://soundcloud.com/user-320158268/sets/pitcher-examples) and [here](https://tinyurl.com/yckcmhb2)

### Installation
```
1. Use git to clone this repo, or download it as a ZIP using the "Clone or download" button & unzip
2. Open your terminal of choice
3. cd to the new pitcher directory
4. pip install -r ./requirements.txt
```

### Usage:
```
python pitcher_cli.py --input-file ./input.wav --st -4 --output-file ./output.wav
```

You can now also run a simple gui version using the command:

```python pitcher_gui.py```


The [releases page](https://github.com/mwcm/pitcher/releases/tag/0.5.2) also has binary files for the GUI (.exe and .app).


### Options:

```
--st                        - number of semitones to shift pitch by,                 int,    required
--input-file                - path to input audio file (WAV, MP3, OGG, FLAC),        string, required
--output-file               - path to output audio file,                             string, required
--log-level                 - logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), string, default 'INFO'
--input-filter              - apply input anti-aliasing filter,                      flag,   default True
--no-input-filter           - skip input anti-aliasing filter
--quantize                  - apply ADC quantization simulation,                     flag,   default True
--no-quantize               - skip ADC quantization simulation
--time-stretch              - enable time stretching,                                flag,   default True
--no-time-stretch           - disable time stretching
--output-filter             - apply output EQ filtering,                             flag,   default True
--no-output-filter          - skip output EQ filtering
--normalize-output          - normalize output audio,                                flag,   default False
--quantize-bits             - bit depth for quantization simulation,                 int,    default 12
--custom-time-stretch       - time stretch factor (1.0=device default, 0.0=none),    float,  default 1.0
--output-filter-type        - output filter: lp1 (7.5kHz), lp2 (10kHz), moog,        str,    default 'lp1'
--moog-output-filter-cutoff - cutoff frequency for moog filter in Hz (20-20000),     int,    default 10000
--force-mono                - convert input to mono (output will also be mono),      flag,   default False
--use-sp12-rate             - use SP-12 SR (27500 Hz) instead of SP-1200 (26040 Hz), flag,   default False
```

### Usage Examples:

```bash
# Basic pitch shifting
python pitcher_cli.py --input-file input.wav --output-file output.wav --st -4

# Disable specific processing steps
python pitcher_cli.py --input-file input.wav --output-file output.wav --st 2 --no-quantize --no-output-filter

# Use moog filter with custom cutoff
python pitcher_cli.py --input-file input.wav --output-file output.wav --st 0 --output-filter-type moog --moog-output-filter-cutoff 5000

# Minimal processing (bypass most effects)
python pitcher_cli.py --input-file input.wav --output-file output.wav --st -1 --no-input-filter --no-time-stretch --no-output-filter
```

If you find this project useful, please consider donating to the [NAACP Legal Defense Fund](https://engage.naacpldf.org/dBCvDTd9IEiXX_jPkmkT_w2) or [BLM CA](https://www.blacklivesmatter.ca/)
