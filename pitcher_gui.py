#!/usr/bin/env python

from tkinter import Button, DoubleVar, IntVar, END, Entry, Entry, filedialog, Label, Scale, Tk, Checkbutton

from pitcher import pitch, OUTPUT_FILTER_TYPES


def gui():
    window = Tk()
    window.geometry('600x400')
    window.resizable(True, False)
    window.title('Pitcher')

    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=3)

    current_st_value = DoubleVar()
    current_bit_value = DoubleVar()

    current_st_value.set(0)
    current_bit_value.set(12)
    
    def get_current_st_value():
        return '{: .2f}'.format(current_st_value.get())

    def get_current_bit_value():
        return '{: .2f}'.format(current_bit_value.get())

    st_slider = Scale(
        window,
        from_= 12,
        to=-12,
        orient='vertical',
        tickinterval=1,
        length=200,
        variable=current_st_value
        )

    bit_slider = Scale(
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

    st_slider_label = Label(
        window,
        text='Semitones:'
    )

    bit_slider_label = Label(
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

    input_entry = Entry(width=60)
    input_entry.grid(column=1, row=4, sticky='w')

    output_entry = Entry(width=60)
    output_entry.grid(column=1, row=5, sticky='w')

    def askopeninputfilename():
        input_file = filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=window, title='Choose a file')
        input_entry.delete(0, END)
        input_entry.insert(0, input_file)

    def askopenoutputfilename():
        output_file = filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=window, title='Choose a file')
        output_entry.delete(0, END)
        output_entry.insert(0, output_file)

    input_browse_button = Button(window, text='Input File', command=askopeninputfilename, width=16)
    input_browse_button.grid(column=0, row=4, sticky='w')

    output_browse_button = Button(window, text='Output File', command=askopenoutputfilename, width=16)
    output_browse_button.grid(column=0, row=5, sticky='w')

    input_filter = IntVar(value=1)
    Checkbutton(window, text="Input Filter", variable=input_filter).grid(column=0, row=6, sticky='w')

    normalize_output = IntVar(value=0)
    Checkbutton(window, text="Normalize Output", variable=normalize_output).grid(column=1, row=6, sticky='w')

    time_stretch = IntVar(value=1)
    Checkbutton(window, text="Time Stretch", variable=time_stretch).grid(column=0, row=7, sticky='w')

    output_filter = IntVar(value=1)
    Checkbutton(window, text="Output Filter", variable=output_filter).grid(column=1, row=7, sticky='w')

    force_mono = IntVar(value=0)
    Checkbutton(window, text="Force Mono", variable=force_mono).grid(column=0, row=8, sticky='w')

    run_button = Button(
        window,
        text='Pitch', 
        command= lambda: pitch(
            st=int(float(get_current_st_value())),
            input_file=input_entry.get(),
            output_file=output_entry.get(),
            log_level='INFO',
            input_filter=True if input_filter else False,
            quantize=bool(input_filter.get()),
            time_stretch=bool(time_stretch.get()),
            normalize_output=bool(normalize_output.get()),
            output_filter=bool(output_filter.get()),
            quantize_bits=int(float(get_current_bit_value())),
            custom_time_stretch=1.0,
            output_filter_type=OUTPUT_FILTER_TYPES[0],
            moog_output_filter_cutoff=10000,
            force_mono=bool(force_mono.get())
        )
    )

    run_button.grid(column=0, padx=5, row=9, sticky='w')
    window.mainloop()


if __name__ == '__main__':
    gui()