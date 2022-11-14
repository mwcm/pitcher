#!/usr/bin/env python

from tkinter import Button, DoubleVar, IntVar, END, Entry, Entry, filedialog, Label, Scale, Tk, Checkbutton, Frame

from pitcher import pitch, OUTPUT_FILTER_TYPES


def gui():
    window = Tk()
    window.geometry('600x400')
    window.resizable(True, False)
    window.title('Pitcher')

    current_st_value = DoubleVar()
    current_bit_value = DoubleVar()

    current_st_value.set(0)
    current_bit_value.set(12)
    
    def get_current_st_value():
        return '{: .2f}'.format(current_st_value.get())

    def get_current_bit_value():
        return '{: .2f}'.format(current_bit_value.get())

    # sliders
    st_frame = Frame(window)
    st_frame.pack(anchor='nw', side='left')

    bit_frame = Frame(window)
    bit_frame.pack(anchor='nw', side='left')

    st_slider_label = Label(st_frame, text='Semitones:')
    st_slider_label.pack(side='top')

    st_slider = Scale(
        st_frame,
        from_= 12,
        to=-12,
        orient='vertical',
        tickinterval=1,
        length=200,
        variable=current_st_value
        )
    st_slider.pack()


    bit_slider_label = Label(bit_frame, text='Quantize Bits:')
    bit_slider_label.pack()

    bit_slider = Scale(
        bit_frame,
        from_= 16,
        to = 2,
        orient='vertical',
        length = 200,
        tickinterval=2,
        variable=current_bit_value
        )
    bit_slider.pack()

    # checkboxes

    input_filter     = IntVar(value=1)
    output_filter    = IntVar(value=1)
    time_stretch     = IntVar(value=1)
    normalize_output = IntVar(value=0)
    force_mono       = IntVar(value=0)

    c_frame = Frame(window)
    c_frame.pack()

    input_filter_button     = Checkbutton(c_frame, text="Input Filter",     variable=input_filter)
    normalize_output_button = Checkbutton(c_frame, text="Normalize Output", variable=normalize_output)
    time_stretch_button     = Checkbutton(c_frame, text="Time Stretch",     variable=time_stretch)
    output_filter_button    = Checkbutton(c_frame, text="Output Filter",    variable=output_filter)
    force_mono_button       = Checkbutton(c_frame, text="Force Mono",       variable=force_mono)

    input_filter_button.pack()
    normalize_output_button.pack()
    time_stretch_button.pack()
    output_filter_button.pack()
    force_mono_button.pack()

    # file input/output

    io_frame = Frame(window)
    io_frame.pack()

    input_entry  = Entry(width=60)
    output_entry = Entry(width=60)

    def askopeninputfilename():
        input_file = filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=io_frame, title='Choose a file')
        input_entry.delete(0, END)
        input_entry.insert(0, input_file)

    # TODO: use asksaveasfilename instead
    def askopenoutputfilename():
        output_file = filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=io_frame, title='Choose a file')
        output_entry.delete(0, END)
        output_entry.insert(0, output_file)

    input_browse_button = Button(window, text='Input File', command=askopeninputfilename, width=16)
    output_browse_button = Button(window, text='Output File', command=askopenoutputfilename, width=16)

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

    run_button
    window.mainloop()


if __name__ == '__main__':
    gui()