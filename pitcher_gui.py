#!/usr/bin/env python

from tkinter import Button as tk_button
from tkinter import DoubleVar as tk_double
from tkinter import END as tk_END
from tkinter import Entry as tk_entry
from tkinter import filedialog as tk_filedialog
from tkinter import Label as tk_label
from tkinter import Scale as tk_scale
from tkinter import Tk

from pitcher import pitch, OUTPUT_FILTER_TYPES


def gui():
    window = Tk()
    window.geometry('600x320')
    window.resizable(True, False)
    window.title('P I T C H E R')

    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=3)

    current_st_value = tk_double()
    current_bit_value = tk_double()

    current_st_value.set(0)
    current_bit_value.set(12)
    
    def get_current_st_value():
        return '{: .2f}'.format(current_st_value.get())

    def get_current_bit_value():
        return '{: .2f}'.format(current_bit_value.get())

    st_slider = tk_scale(
        window,
        from_= 12,
        to=-12,
        orient='vertical',
        tickinterval=1,
        length=200,
        variable=current_st_value
        )

    bit_slider = tk_scale(
        window,
        from_= 16,
        to = 2,
        orient='vertical',
        length = 200,
        tickinterval=2,
        variable=current_bit_value
        )

    st_slider.grid(
        column=0,
        padx=5,
        row=1,
        sticky='w'
    )

    bit_slider.grid(
        column=1,
        padx=5,
        row=1,
        sticky='w'
    )

    st_slider_label = tk_label(
        window,
        text='Semitones:'
    )

    bit_slider_label = tk_label(
        window,
        text='Quantize Bits:'
    )

    st_slider_label.grid(
        column=0,
        padx=5,
        row=0,
        sticky='w'
    )

    bit_slider_label.grid(
        column=1,
        padx=5,
        row=0,
        sticky='w'
    )

    input_entry = tk_entry(width=40)
    input_entry.grid(column=1, row=4, sticky='w')

    output_entry = tk_entry(width=40)
    output_entry.grid(column=1, row=5, sticky='w')

    def askopeninputfilename():
        input_file = tk_filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=window, title='Choose a file')
        input_entry.delete(0, tk_END)
        input_entry.insert(0, input_file)

    def askopenoutputfilename():
        output_file = tk_filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=window, title='Choose a file')
        output_entry.delete(0, tk_END)
        output_entry.insert(0, output_file)

    input_browse_button = tk_button(window, text='Input File', command=askopeninputfilename, width=16)
    input_browse_button.grid(column=0, padx=5, row=4, sticky='w')

    output_browse_button = tk_button(window, text='Output File', command=askopenoutputfilename, width=16)
    output_browse_button.grid(column=0, padx=5, row=5, sticky='w')

    run_button = tk_button(
        window,
        text='Pitch', 
        command= lambda: pitch(
            st=int(float(get_current_st_value())),
            input_file=input_entry.get(),
            output_file=output_entry.get(),
            log_level='INFO',
            input_filter=True,
            quantize=True,
            time_stretch=True,
            normalize_output=False,
            output_filter=True,
            quantize_bits=int(float(get_current_bit_value())),
            custom_time_stretch=1.0,
            output_filter_type=OUTPUT_FILTER_TYPES[0],
            moog_output_filter_cutoff=10000
        )
    )

    run_button.grid(column=0, padx=5, row=6, sticky='w')
    window.mainloop()



if __name__ == '__main__':
    gui()