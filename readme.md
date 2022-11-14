# Pitcher.py
<img src="https://user-images.githubusercontent.com/2433319/130370952-3b029cf5-d9b7-4877-be0b-8593c017b5ea.png" width="600" height="320">

- Free & OS emulation of the SP-12 & SP-1200 signal chain (now with GUI)
- Pitch shift / bitcrush / resample audio files
- Written and tested in Python v3.10.7
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


The [releases page](https://github.com/mwcm/pitcher/releases/tag/0.0.1) also has binary files for the GUI (.exe and .app).


### Options:
```
--st                        - number of semitones to shift pitch by,                 int,    required
--input-file                - path to input file,                                    string, required
--output-file               - path to output file,                                   string, required
--log-level                 - sets logging threshold,                                string, default 'INFO'
--input-filter              - input anti aliasing low pass filter,                   flag,   default True
--quantize                  - simulate ADC quantize,                                 flag,   default True
--time-stretch              - enable or disable time_shift entirely,                 flag,   default True
--output-filter             - skip all output filtering (default and moog),          flag,   default True
--normalize-output          - normalize output volume to ,                           flag,   default False
--quantize-bits             - bit rate of quantized output,                          int,    default 12
--custom-time-stretch       - custom shift, 1.0 for device default, 0.0 for none,    float,  default 1.0
--output-filter-type        - LP or MoogLP,                                          str,    default 'LP'
--moog-output-filter-cutoff - set cutoff for moog SSM2044 approximation,             int,    default 10000
--force-mono                - convert input to mono, ouput will also be mono,        flag,   default False
```

If you find this project useful, please consider donating to the [NAACP Legal Defense Fund](https://org2.salsalabs.com/o/6857/p/salsa/donation/common/public/?donate_page_KEY=15780&_ga=2.209233111.496632409.1590767838-1184367471.1590767838) or [BLM - TO](https://blacklivesmatter.ca/donate/)


### Current TODO:
- add all options to GUI
- scipy resample vs librosa resample and usages (git history? comments?)

### Future TODO:
- smaller exe size
- dedicated 33rpm -> 45rpm pre-processing option
- only use ffmpeg/libav when necessary
- perfect high end input anti aliasing filter fit (close enough, not a priority for now)
