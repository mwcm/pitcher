# pitcher

Emulation of the SP-12 & SP-1200 signal chain in a Python script

Based on: [Physical and Behavioral Circuit Modeling of the SP-12
Sampler](https://ccrma.stanford.edu/~dtyeh/sp12/yeh2007icmcsp12slides.pdf)

Useful for re-sampling & pitching audio to get that "SP-1200 sound" before importing into a DAW

### usage:
```
python3 pitcher.py --file ./input.wav --st -4 --output-file ./output.wav
```

### options:
```
--st                 - # of semitones to shift the pitch by, int, required
--normalize          - optionally normalize output audio, flag
--input-file         - path to input file, string, required
--output-file        - path to output file, string, required
--resample-fn        - resample method to use, string, 'scipy' or 'librosa',
                       default 'scipy'(WIP likely to change soon)
--skip-quantize      - optionally skip ADC quantize step, flag
--skip-input-filter  - optionally skip input anti aliasing filter, flag
--skip-output-filter - optionally skip output equalization filter, flag
```
