# Pitcher.py
- Emulation of the SP-12 & SP-1200 signal chain
- Written and tested in Python v3.7.7
- Based on [Physical and Behavioral Circuit Modeling of the SP-12
Sampler, DT Yeh, 2007](https://ccrma.stanford.edu/~dtyeh/papers/yeh07_icmc_sp12.pdf) & [Slides](https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf)
- Audio examples [here](https://soundcloud.com/user-320158268/sets/pitcher-examples) and [here](https://tinyurl.com/yckcmhb2)

### Installation
```
1. Use git to clone this repo, or download it as a ZIP using the "Clone or download" button above & unzip
2. Open your terminal of choice
3. cd to the new pitcher directory
4. pip3 install -r ./requirements.txt
```

### Usage:
```
python3 pitcher.py --file ./input.wav --st -4 --output-file ./output.wav
```

### Options:
```
--st                 - # semitones to shift pitch by,   int,    required
--input-file         - path to input file,              string, required
--output-file        - path to output file,             string, required
--quantize-bits      - bit rate of quantized output,    int,    default 12
--skip-quantize      - skip simulation of ADC quantize, flag
--skip-normalize     - skip output normalization,       flag
--skip-input-filter  - skip input anti aliasing filter, flag
--skip-output-filter - skip output equalization filter, flag
```

### TODO:
- logging
- fix saving to mp3 issue, fork audiofile
- optionally preserve stereo channels throughout processing
- optional vcf (moog ring) good description in slides
- time_shift/no time_shift option
- replace librosa if there is a module with better performance, maybe essentia?
- improve high end input anti aliasing filter fit?
