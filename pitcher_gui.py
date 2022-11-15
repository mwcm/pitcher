#!/usr/bin/env python

from tkinter import (
    Button, DoubleVar, IntVar, END, Entry, Entry, filedialog, Label, Scale, Tk, 
    Checkbutton, Frame, StringVar, OptionMenu
    )

from pitcher import pitch, OUTPUT_FILTER_TYPES


def gui():
    window = Tk()
    window.geometry('600x400')
    window.resizable(True, False)
    window.title('Pitcher')

    current_st_value = DoubleVar()
    current_bit_value = DoubleVar()
    current_time_stretch_value = DoubleVar()

    current_st_value.set(0)
    current_bit_value.set(12)
    current_time_stretch_value.set(1.0)
    
    def get_current_st_value():
        return int(current_st_value.get())

    def get_current_bit_value():
        return int(current_bit_value.get())

    def get_current_time_stretch_value():
        return current_time_stretch_value.get()
    
    top_frame = Frame(window)
    middle_frame = Frame(window)

    top_frame.pack(side='top', fill='x')
    middle_frame.pack(side='top', fill='x')

    # sliders
    st_frame = Frame(top_frame)
    st_frame.pack(anchor='nw', side='left', padx=10, pady=10)

    bit_frame = Frame(top_frame)
    bit_frame.pack(anchor='nw', side='left', padx=10, pady=10)

    stretch_frame = Frame(top_frame)
    stretch_frame.pack(anchor='nw', side='left', padx=10, pady=10)

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
    bit_slider_label.pack(side='top')

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

    stretch_slider_label = Label(stretch_frame, text='Time Stretch:')
    stretch_slider_label.pack(side='top')

    stretch_slider = Scale(
        stretch_frame,
        from_= 5,
        to = 0.05,
        orient='vertical',
        length = 200,
        resolution=0.01,
        variable=current_time_stretch_value
        )
    stretch_slider.pack()

    # other options

    input_filter     = IntVar(value=1)
    quantize         = IntVar(value=1)
    output_filter    = IntVar(value=1)
    time_stretch     = IntVar(value=1)
    normalize_output = IntVar(value=0)
    force_mono       = IntVar(value=0)
    output_filter_type = StringVar(value=OUTPUT_FILTER_TYPES[0])

    o_frame = Frame(top_frame)
    o_frame.pack(padx=20, pady=10)

    input_filter_button     = Checkbutton(o_frame, text="Input Filter",     variable=input_filter)
    quantize_button         = Checkbutton(o_frame, text="Quantize",         variable=quantize)
    normalize_output_button = Checkbutton(o_frame, text="Normalize Output", variable=normalize_output)
    time_stretch_button     = Checkbutton(o_frame, text="Time Stretch",     variable=time_stretch)
    output_filter_button    = Checkbutton(o_frame, text="Output Filter",    variable=output_filter)
    force_mono_button       = Checkbutton(o_frame, text="Force Mono",       variable=force_mono)

    oft_label = Label(o_frame, text='Output Filter Type:')
    output_filter_menu = OptionMenu(o_frame, output_filter_type, *OUTPUT_FILTER_TYPES)

    oft_label.pack()
    output_filter_menu.pack()

    input_filter_button.pack()
    quantize_button.pack()
    normalize_output_button.pack()
    time_stretch_button.pack()
    output_filter_button.pack()
    force_mono_button.pack()

    # file input/output

    io_frame = Frame(middle_frame)
    io_frame.pack(fill='x')

    i_frame = Frame(io_frame)
    o_frame = Frame(io_frame)

    i_frame.pack(fill='both', side='top', padx=20, pady=10)
    o_frame.pack(fill='x', padx=20, pady=10)

    input_entry  = Entry(i_frame, width=60)
    output_entry = Entry(o_frame, width=60)

    def askopeninputfilename():
        input_file = filedialog.askopenfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=i_frame, title='Choose a file')
        input_entry.delete(0, END)
        input_entry.insert(0, input_file)

    def askopenoutputfilename():
        output_file = filedialog.asksaveasfilename(filetypes=[("audio files", "*.mp3 *.wav *.flac")], parent=o_frame, title='Choose a file')
        output_entry.delete(0, END)
        output_entry.insert(0, output_file)

    input_browse_button  = Button(i_frame, text='Input File', command=askopeninputfilename, width=16)
    output_browse_button = Button(o_frame, text='Output File', command=askopenoutputfilename, width=16)

    input_browse_button.pack(side='left')
    input_entry.pack(side='top')

    output_browse_button.pack(side='left')
    output_entry.pack()

    run_button = Button(
        window,
        text='Pitch', 
        command= lambda: pitch(
            st=get_current_st_value(),
            input_file=input_entry.get(),
            output_file=output_entry.get(),
            log_level='INFO',
            input_filter=bool(input_filter.get()),
            quantize=bool(quantize.get()),
            time_stretch=bool(time_stretch.get()),
            normalize_output=bool(normalize_output.get()),
            output_filter=bool(output_filter.get()),
            quantize_bits=get_current_bit_value(),
            custom_time_stretch=get_current_time_stretch_value(),
            output_filter_type=output_filter_type.get(),
            moog_output_filter_cutoff=10000,
            force_mono=bool(force_mono.get())
        )
    )

    run_button.pack()
    window.mainloop()


if __name__ == '__main__':
    gui()