import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from itertools import combinations

Spectrum = np.array([1, 2, 3, 4, 5])  # 設定預設值
Standard_Wavelength = np.array([365.34, 406.15, 436.00, 545.79, 578.60, 696.56, 706.58, 727.17,
                                738.34, 750.66, 763.56, 772.34, 794.56, 800.98, 811.48, 826.63,
                                842.33, 852.20, 866.79, 912.38, 922.18])

def read_numeric_data(filename):
    data = []
    has_header = False
    intensity_index = 0  # Default to the first column

    with open(filename, 'r') as file:
        first_line = file.readline().strip().split()
        if any(element.isalpha() for element in first_line):
            has_header = True
            if "Intensity" in first_line:
                intensity_index = first_line.index("Intensity")
            else:
                raise ValueError("No column labeled 'Intensity' found in header.")
        if has_header:
            for line in file:
                elements = line.strip().split()
                data.append(float(elements[intensity_index]))
        else:
            data.append(float(first_line[intensity_index]))
            for line in file:
                elements = line.strip().split()
                data.append(float(elements[0]))

    return np.array(data)

def open_image():
    img_path = "HgArSpectralExample.png"
    img = Image.open(img_path)
    img.show()

def calculate_fwhm(spectrum, width, prominence, height):
    try:
        height_value = float(height)
    except ValueError:
        height_value = None

    peaks, _ = find_peaks(spectrum[0:1240], width=width, prominence=prominence, height=height_value, distance=6)
    fwhm_values = []
    for peak in peaks:
        peak_height = spectrum[peak]
        half_height = peak_height / 2.0
        left_idx = peak - np.argmax(spectrum[peak::-1] <= half_height)
        right_idx = peak + np.argmax(spectrum[peak:] <= half_height)
        fwhm = round((right_idx - left_idx), 1)
        fwhm_values.append(fwhm)

    return peaks, fwhm_values

def open_file():
    global Spectrum
    filename = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if filename:
        Spectrum = read_numeric_data(filename)
        Spectrum = Spectrum - np.average(Spectrum[0:50])
        update_plot_and_results(Spectrum)

def perform_fitting():
    global Standard_Wavelength, Spectrum

    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)
    height = round(float(height_entry.get()), 1)
    peaks, fwhm_values = calculate_fwhm(Spectrum, width, prominence, height)

    switch_vars = [var.get() for var in switch_vars1 + switch_vars2]

    if len(peaks) < sum(switch_vars):
        fitting_label.config(text="峰值數量不足，無法進行拟合")
        correlation_label.config(text="")
        return

    selected_standard_peak = [peak for peak, switch in zip(Standard_Wavelength, switch_vars) if switch != 0]
    Standard_peak = selected_standard_peak
    Spectrum_peak = peaks[:len(selected_standard_peak)]

    coefficients = np.polyfit(Spectrum_peak, Standard_peak, 2)
    f = np.poly1d(coefficients)

    correlation = np.corrcoef(Standard_peak, f(Spectrum_peak))[0, 1]

    fitting_label.config(text=f"Fitting Curve Equation: {f}")
    correlation_label.config(text=f"Correlation Coefficient: {correlation:.4f}")

    ax_fit.cla()
    ax_fit.plot(Spectrum_peak, Standard_peak, 'ro', label='Points', ms=3)
    ax_fit.plot(Spectrum_peak, f(Spectrum_peak), 'b--', label='Fitted Curve')
    ax_fit.set_xlabel('Standard_peak (nm)')
    ax_fit.set_ylabel('Spectrum_peak (pixel)')
    ax_fit.legend()
    canvas_fit.draw()

    FWHM_convert = [(coefficients[1] + coefficients[0] * peak * 2) * fwhm for peak, fwhm in zip(peaks, fwhm_values)]

    update_table_with_fwhm_nm(results_table, peaks, fwhm_values, FWHM_convert)

    return FWHM_convert

def update_table_with_fwhm_nm(table, peaks, fwhm_values, FWHM_convert):
    for i in table.get_children():
        table.delete(i)

    for peak, fwhm, fwhm_nm in zip(peaks, fwhm_values, FWHM_convert):
        table.insert('', 'end', values=(peak, fwhm, round(fwhm_nm, 1)))

def create_table(parent):
    tree = ttk.Treeview(parent, columns=('Peak Position', 'Wavelength', 'FWHM(nm)'), show='headings')
    scrollbar = ttk.Scrollbar(parent, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    tree.heading('Peak Position', text='峰值位置')
    tree.heading('Wavelength', text='Wavelength\n(nm)')
    tree.heading('FWHM(nm)', text='FWHM\n(nm)')

    tree.column('Peak Position', anchor='center', width=60)
    tree.column('Wavelength', anchor='center', width=60)
    tree.column('FWHM(nm)', anchor='center', width=60)

    tree.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    return tree

def update_table(table, peaks, fwhm_values):
    for i in table.get_children():
        table.delete(i)

    for peak, fwhm in zip(peaks, fwhm_values):
        table.insert('', 'end', values=(peak, fwhm))

def update_plot_and_results(Spectrum):
    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)

    peaks, fwhm_values = calculate_fwhm(Spectrum, width, prominence, height_entry.get())

    ax.cla()
    ax.plot(Spectrum)
    ax.plot(peaks, Spectrum[peaks], 'ro')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity')
    canvas.draw()

    update_table(results_table, peaks, fwhm_values)

def update_width(event):
    update_plot_and_results(Spectrum)

def update_prominence(event):
    update_plot_and_results(Spectrum)

def terminate_program():
    sys.exit()

def cubic_fitting():
    global Standard_Wavelength, Spectrum

    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)
    height = round(float(height_entry.get()), 1)
    peaks, fwhm_values = calculate_fwhm(Spectrum, width, prominence, height)

    if len(peaks) > len(Standard_Wavelength):
        fitting_label.config(text="峰值數量大於數列，無法進行擬合")
        correlation_label.config(text="")
        return

    best_r2 = -np.inf
    best_fit = None
    best_coefficients = None
    best_comb = None

    for comb in combinations(Standard_Wavelength, len(peaks)):
        coefficients = np.polyfit(peaks, comb, 2)
        f = np.poly1d(coefficients)
        r2 = np.corrcoef(comb, f(peaks))[0, 1] ** 2

        if r2 > best_r2:
            best_fit = f
            best_r2 = r2
            best_coefficients = coefficients
            best_comb = comb

    if best_r2 > 0.9995:
        fitting_label.config(
            text=f"Fitting Curve Equation: {best_fit}\nR²: {best_r2:.4f}\nCoefficients: {best_coefficients}")
    else:
        fitting_label.config(text="無法達到 R2=0.9995 的擬合")

    ax_fit.cla()
    ax_fit.plot(peaks, best_comb, 'ro', label='Points', ms=3)
    if best_fit:
        ax_fit.plot(peaks, best_fit(peaks), 'b--', label='Fitted Curve')
    ax_fit.set_xlabel('Peak Position (pixel)')
    ax_fit.set_ylabel('Standard Wavelength (nm)')
    ax_fit.legend()
    canvas_fit.draw()

    FWHM_convert = [(best_coefficients[2] + best_coefficients[1] * peak * 2 + 3 * best_coefficients[0] * peak ** 2) * fwhm for peak, fwhm in zip(peaks, fwhm_values)]
    update_table_with_fwhm_nm(results_table, peaks, best_comb, FWHM_convert)

root = tk.Tk()
root.title("可讀性程式")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

left_frame = tk.Frame(root)
left_frame.grid(row=0, column=0, sticky="nsew")
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)
results_table = create_table(left_frame)

right_frame = tk.Frame(root)
right_frame.grid(row=0, column=1, sticky="nsew")
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

result_label = tk.Label(left_frame, text="")
result_label.pack()

scale_frame = tk.Frame(left_frame)
scale_frame.pack()

width_label = tk.Label(scale_frame, text="Width: ")
width_label.pack(side=tk.LEFT)
width_scale = tk.Scale(scale_frame, from_=0, to=10, resolution=0.1, length=150, orient=tk.HORIZONTAL, command=update_width)
width_scale.set(1.5)
width_scale.pack(side=tk.LEFT, padx=5)

prominence_label = tk.Label(scale_frame, text="Prominence: ")
prominence_label.pack(side=tk.LEFT)
prominence_scale = tk.Scale(scale_frame, from_=0, to=20, resolution=0.1, length=150, orient=tk.HORIZONTAL, command=update_prominence)
prominence_scale.set(6)
prominence_scale.pack(side=tk.LEFT, padx=5)

height_label = tk.Label(scale_frame, text="Height: ")
height_label.pack(side=tk.LEFT, pady=5)

height_var = tk.StringVar()
height_var.set("250")

height_entry = tk.Entry(scale_frame, textvariable=height_var, width=5)
height_entry.pack(side=tk.LEFT, padx=5, pady=5)

switch_frame = tk.Frame(left_frame)
switch_frame.pack()

names = ['365', '405', '436', '545', '578', '696', '706', '727', '738', '750', '763', '772', '794', '800', '811', '826',
         '842', '852', '866', '912', '922']

names1 = names[:11]
names2 = names[11:]

switch_frame1 = tk.Frame(switch_frame)
switch_frame1.pack(side=tk.TOP, padx=10, pady=10)

switch_frame2 = tk.Frame(switch_frame)
switch_frame2.pack(side=tk.BOTTOM, padx=10, pady=10)

switch_vars1 = []
switches1 = []
for name in names1:
    var = tk.IntVar()
    var.set(1)
    switch_vars1.append(var)
    switch = tk.Checkbutton(switch_frame1, text=name, variable=var)
    switch.pack(side=tk.LEFT)
    switches1.append(switch)

switch_vars2 = []
switches2 = []
for name in names2:
    var = tk.IntVar()
    var.set(1)
    switch_vars2.append(var)
    switch = tk.Checkbutton(switch_frame2, text=name, variable=var)
    switch.pack(side=tk.LEFT)
    switches2.append(switch)

button_frame = tk.Frame(left_frame)
button_frame.pack()

open_button = tk.Button(button_frame, text='Open', command=open_file)
open_button.pack(side=tk.LEFT, padx=5)

fitting_button = tk.Button(button_frame, text='Fitting', command=perform_fitting)
fitting_button.pack(side=tk.LEFT, padx=5)

image_button = tk.Button(button_frame, text='Open Image', command=open_image)
image_button.pack(side=tk.LEFT, padx=5)

terminate_button = tk.Button(button_frame, text='Terminate', command=terminate_program)
terminate_button.pack(side=tk.LEFT, padx=5)

cubic_fitting_button = tk.Button(button_frame, text='Cubic Fitting', command=cubic_fitting)
cubic_fitting_button.pack(side=tk.LEFT, padx=5)

fig_fit, ax_fit = plt.subplots()
canvas_fit = FigureCanvasTkAgg(fig_fit, master=right_frame)
canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)

fitting_label = tk.Label(right_frame, text="")
fitting_label.pack()

correlation_label = tk.Label(right_frame, text="")
correlation_label.pack()

def update_correlation_label(correlation):
    correlation_label.config(text="相關係數: {:.4f}".format(correlation))

root.mainloop()
