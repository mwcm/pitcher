# Pitcher.py
<img src="https://user-images.githubusercontent.com/2433319/130370952-3b029cf5-d9b7-4877-be0b-8593c017b5ea.png" width="600" height="320">

- Free & OS emulation of the SP-12 & SP-1200 signal chain (now with GUI)
- Pitch shift / bitcrush / resample audio files
- Written and tested in Python v3.7.7
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
python pitcher.py --input-file ./input.wav --st -4 --output-file ./output.wav
```

You can now also run a simple gui version using the command:

```python pitcher_gui.py```

For Mac Users:

```python pitcher_gui_mac.py```

The [releases page](https://github.com/mwcm/pitcher/releases/tag/0.0.1) also has binary files for the GUI (.exe and .app).


### Options:
```
--st                 - # semitones to shift pitch by,   		            int,    required
--input-file         - path to input file,              		            string, required
--output-file        - path to output file,             		            string, required
--quantize-bits      - bit rate of quantized output,    		            int,    default 12
--time-shift         - custom time shift ratio to apply,		            float,  default 0
--skip-quantize      - skip simulation of ADC quantize, 		            flag
--skip-normalize     - skip output normalization,       		            flag
--skip-input-filter  - skip input anti aliasing filter, 		            flag
--skip-output-filter - skip output equalization LP filter,              flag
--skip-time-shift    - skip time shift inherent to pitching algorithm,  flag
--moog-filter        - enable Moog LP output filter, emulates SSM2044,  flag
```

If you find this project useful, please consider donating to the [NAACP Legal Defense Fund](https://org2.salsalabs.com/o/6857/p/salsa/donation/common/public/?donate_page_KEY=15780&_ga=2.209233111.496632409.1590767838-1184367471.1590767838) or [BLM - TO](https://blacklivesmatter.ca/donate/)


### TODO:
- enable console window on mac gui too (already defaults to shown in windows gui)
- enable moog filter for gui versions
- combine pitcher_gui.py and pitcher_gui_mac.py
- combine pitcher and pitcher_gui.py
- add all options to GUI
- dedicated 33rpm -> 45rpm pre-processing option, add to GUI
- only use ffmpeg/libav when necessary
- optionally preserve stereo channels throughout processing
- replace librosa if there is a module with better performance, maybe essentia?
- perfect high end input anti aliasing filter fit, likely not very important
