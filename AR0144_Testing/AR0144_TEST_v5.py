import tkinter as tk
from tkinter import ttk, filedialog, TclError
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from itertools import combinations
import ctypes
from ctypes import c_void_p, c_int, byref, create_string_buffer, wintypes
import json
import os

# Functions and variables from the second script
Standard_Wavelength = np.array([365.34, 406.15, 436.00, 545.79, 578.60, 696.56, 706.58, 727.17,
                                738.34, 750.66, 763.56, 772.34, 794.56, 800.98, 811.48, 826.63,
                                842.33, 852.20, 866.79, 912.38, 922.18])


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.rows_number = 10  # Initial number of rows
        self.brightness = 75  # Initial Brightness value
        self.gain = 2  # Initial Gain value
        self.baseline_correction = tk.BooleanVar(value=True)  # Initial baseline correction value
        self.dll = ctypes.WinDLL('Dll/SpectroChipsControl.dll')
        self.initialize_functions()

        self.x_axis_min_var = tk.DoubleVar(value=0)  # Default min value of X-axis
        self.x_axis_max_var = tk.DoubleVar(value=1280)  # Default max value of X-axis

        self.load_config()

        # Initialize the camera
        if self.initialize_device():
            self.cap = self.find_and_initialize_camera()
            if not self.cap:
                raise ValueError("No suitable camera found")
            else:
                self.read_data_from_device()
        else:
            raise ValueError("No suitable camera found")

        # Set camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        self.cap.set(cv2.CAP_PROP_MODE, 2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        # Create GUI layout
        self.create_widgets()

        # Timer
        self.delay = 50
        self.update()

        self.window.mainloop()

    def load_config(self):
        config_file = "config.txt"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as file:
                    config = json.load(file)
                    self.x_axis_min_var.set(config.get("x_axis_min", 1))
                    self.x_axis_max_var.set(config.get("x_axis_max", 1280))
                    self.ROI = config.get("ROI", 470)
                    self.rows_number = config.get("rows_number", 10)
                    self.brightness = config.get("brightness", 75)
                    self.gain = config.get("gain", 2)
                    self.baseline_correction.set(config.get("baseline_correction", True))
                    print("Config loaded successfully")
            except Exception as e:
                print(f"Failed to load config: {e}")
        else:
            print("Config file not found, using default settings")

    def initialize_functions(self):
        # Define function prototypes
        self.SP_Initialize = self.dll.SP_Initialize
        self.SP_Initialize.argtypes = [ctypes.c_void_p]
        self.SP_Initialize.restype = wintypes.DWORD

        self.SP_Finalize = self.dll.SP_Finalize
        self.SP_Finalize.argtypes = [ctypes.c_void_p]
        self.SP_Finalize.restype = wintypes.DWORD

        self.SP_DataRead = self.dll.SP_DataRead
        self.SP_DataRead.argtypes = [c_void_p, ctypes.POINTER(c_int)]
        self.SP_DataRead.restype = ctypes.c_long

    def initialize_device(self):
        # Initialize device
        hr = self.SP_Initialize(None)
        if hr != 0:  # ERROR_SUCCESS is 0
            print("Device initialize fail")
            return False
        print("Device initialized successfully")
        return True
    def read_data_from_device(self):
        # Allocate buffer
        buffer_size = 4096  # 4KB
        buffer = create_string_buffer(buffer_size)
        data_length = c_int(buffer_size)

        # Call the SP_DataRead function
        result = self.SP_DataRead(buffer, byref(data_length))

        # Check the result and process
        if result == 0:  # Assuming 0 indicates success
            # Use the raw attribute to read all data
            read_data = buffer.raw[:data_length.value]
            # Remove null bytes and trailing invalid bytes
            cleaned_data = read_data.replace(b'\x00', b'').rstrip(b'\xff')
            final_string = cleaned_data.decode('utf-8')
            print(f'Cleaned Data: {final_string}')

            # Parse JSON data
            try:
                json_data = json.loads(final_string)
                roi = json_data.get("ROI")
                if roi:
                    print(f'ROI parameter: {roi}')
                    self.ROI = int(roi[2])  # Initial ROI value
                    self.WL = [float(x) for x in json_data.get("WL")]
                    self.x_axis_wavelength = [self.WL[0] + x * self.WL[1] + x * x * self.WL[2] + x * x * x * self.WL[3] for x in range(1280)]
                    self.x_axis_pixel = list(range(1280))
                else:
                    print("ROI parameter not found in JSON data. Using default settings.")
                    self.use_default_settings()
            except json.JSONDecodeError as e:
                print(f'JSONDecodeError: {e}. Using default settings.')
                self.use_default_settings()
        else:
            print(f'Read Failed with error code: {result}')

    def use_default_settings(self):
        # Apply default settings
        self.ROI = 470
        self.x_axis_wavelength = [176.1+ x * 0.6378 + x * x * 1.515e-5 for x in range(1280)]
        self.x_axis_pixel = list(range(1280))
        print("Default settings applied due to an error.")

    def finalize_device(self):
        # Finalize device
        hr = self.SP_Finalize(None)
        if hr != 0:
            print("Device finalize fail")
        else:
            print("Device finalized successfully")

    def find_and_initialize_camera(self):
        # Check for available cameras and their resolutions
        for i in range(10):  # Assuming a maximum of 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if width == 1280 and height == 800:
                    print(f"Camera {i} initialized with resolution 1280x800")
                    return cap
                cap.release()
        return None

    def create_widgets(self):
        # Create buttons
        self.buttons_frame = tk.Frame(self.window)
        self.buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.setting_button = ttk.Button(self.buttons_frame, text="Setting", command=self.open_setting)
        self.setting_button.pack(side=tk.LEFT, padx=2)

        self.save_button = ttk.Button(self.buttons_frame, text="Save", command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=2)

        self.cubic_fitting_button = ttk.Button(self.buttons_frame, text="Cubic Fitting",
                                               command=self.perform_cubic_fitting)
        self.cubic_fitting_button.pack(side=tk.LEFT, padx=2)

        self.exit_button = ttk.Button(self.buttons_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(side=tk.LEFT, padx=2)

        # Baseline correction toggle
        self.baseline_toggle = ttk.Checkbutton(self.buttons_frame, text="Baseline Correction",
                                               variable=self.baseline_correction)
        self.baseline_toggle.pack(side=tk.LEFT, padx=2)

        # Create left side frame (raw camera feed)
        self.camera_frame = ttk.Label(self.window)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=12)

        # Create right side frame (ROI analysis result)
        self.plot_frame = tk.Frame(self.window)
        self.plot_frame.grid(row=0, column=1, padx=14, pady=16)

        # X-axis option
        tk.Label(self.buttons_frame, text="X-axis:").pack(side=tk.LEFT, padx=1)
        self.x_axis_option = ttk.Combobox(self.buttons_frame, values=["Pixel", "Wavelength"], state="readonly")
        self.x_axis_option.current(0)
        self.x_axis_option.pack(side=tk.LEFT, padx=2)
        self.x_axis_option.bind("<<ComboboxSelected>>", self.update_plot_x_axis)

        # Initialize Matplotlib figure and axes for plotting
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill=tk.BOTH, expand=True)

    def update(self):
        # Read frame from camera
        ret, frame = self.cap.read()
        # If successful, update left side frame
        if ret:
            a = frame[0][::2].astype(np.uint16)  # Start from 0, take every second element
            b = frame[0][1::2].astype(np.uint16)  # Start from 1, take every second element
            if len(frame[0]) % 2 != 0:
                b = np.append(b, 0)  # Ensure a and b have the same length

            combined = ((a << 4) | (b - 128)).astype(np.uint16)
            result = combined.reshape(800, 1280).astype(np.float64)
            normalized_result = (result / 4096 * 254).astype(np.uint8)

            start_point = (0, self.ROI - self.rows_number)  # Start point coordinates
            end_point = (normalized_result.shape[1], self.ROI + self.rows_number)  # End point coordinates
            color = (255, 0, 0)  # BGR color value, red
            thickness = 2  # Line thickness

            # Draw red rectangle on resized_image
            cv2.rectangle(normalized_result, start_point, end_point, color, thickness)
            resized_image = cv2.resize(normalized_result, (0, 0), fx=0.4, fy=0.5)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            self.camera_frame.config(image=self.photo)
            self.analyze_roi(result, self.ROI, self.rows_number)
        self.window.after(self.delay, self.update)

    def calculate_fwhm(self, spectrum, width, prominence, height, baseline_correction):
        try:
            height_value = float(height)
        except ValueError:
            height_value = None

        if baseline_correction:
            spectrum = spectrum - np.average(spectrum[0:100])

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

    def update_table_with_fwhm_nm(self, table, peaks, fwhm_values, FWHM_convert):
        for i in table.get_children():
            table.delete(i)

        for peak, fwhm, fwhm_nm in zip(peaks, fwhm_values, FWHM_convert):
            table.insert('', 'end', values=(peak, fwhm, round(fwhm_nm, 1)))

    def create_table(self, parent):
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

    def cubic_fitting(self, roi_avg, baseline_correction):
        global Standard_Wavelength

        width = 4
        prominence = 7.0
        height = 250
        peaks, fwhm_values = self.calculate_fwhm(roi_avg, width, prominence, height, baseline_correction)

        if len(peaks) > len(Standard_Wavelength):
            print("峰值數量大於數列，無法進行擬合")
            return

        best_r2 = -np.inf
        best_fit = None
        best_coefficients = None
        best_comb = None
        count = 0

        for comb in combinations(Standard_Wavelength, len(peaks)):
            coefficients = np.polyfit(peaks, comb, 2)
            f = np.poly1d(coefficients)
            r2 = np.corrcoef(comb, f(peaks))[0, 1] ** 2
            count += 1

            if r2 > best_r2:
                best_fit = f
                best_r2 = r2
                best_coefficients = coefficients
                best_comb = comb

            if best_r2 == 1:
                break

            if count % 15000 == 0:
                print(f"Tested {count} combinations so far...")

        if best_r2 > 0.9997:
            result_text = f"Fitting Curve Equation: {best_fit}\nR²: {best_r2:.4f}\nCoefficients: {best_coefficients}"
            print(f"Final count{count}")
        else:
            result_text = "無法達到 R2=0.9995 的擬合"

        # Create a new window to display -results
        result_window = tk.Toplevel()
        result_window.title("Cubic Fitting Results")

        fig_fit, ax_fit = plt.subplots()
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=result_window)
        canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ax_fit.cla()
        ax_fit.plot(peaks, best_comb, 'ro', label='Points', ms=3)
        if best_fit:
            ax_fit.plot(peaks, best_fit(peaks), 'b--', label='Fitted Curve')
        ax_fit.set_xlabel('Peak Position (pixel)')
        ax_fit.set_ylabel('Standard Wavelength (nm)')
        ax_fit.legend()
        canvas_fit.draw()

        # Display the fitting results in the new window
        fitting_label = tk.Label(result_window, text=result_text)
        fitting_label.pack()
        # Create and update the FWHM table
        fwhm_table = self.create_table(result_window)
        FWHM_convert = [(best_coefficients[1] + best_coefficients[0] * peak * 2) * fwhm for peak, fwhm in
                        zip(peaks, fwhm_values)]
        self.update_table_with_fwhm_nm(fwhm_table, peaks, best_comb, FWHM_convert)

    def analyze_roi(self, result, ROI, rows_number):
        roi = result[ROI - rows_number: ROI + rows_number, :]
        roi_avg = np.mean(roi, axis=0)
        self.roi_avg = roi_avg  # Store roi_avg for cubic fitting
        self.draw_plot(roi_avg)

    def draw_plot(self, data):
        # Clear the previous data
        self.ax.clear()

        # Plot new data
        if self.baseline_correction.get():
            data = data - np.average(data[0:100])

        x_axis = self.x_axis_wavelength if self.x_axis_option.get() == "Wavelength" else self.x_axis_pixel
        self.ax.plot(x_axis, data)

        x_min = self.safe_get_float(self.x_axis_min_var, 1)  # 默认为 1
        x_max = self.safe_get_float(self.x_axis_max_var, 1280)  # 默认为 1280
        self.ax.set_xlim([x_min, x_max])  # Apply the new X-axis limits

        self.ax.set_ylim([0, 4100])  # Set y-axis limits
        self.ax.set_title("Real-time Spectrum")
        xlabel = "Wavelength (nm)" if self.x_axis_option.get() == "Wavelength" else "Wavelength (pixel)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Intensity (a.u.)")

        self.canvas.draw()

    def update_plot_x_axis(self, event):
        self.draw_plot(self.roi_avg)

    def perform_cubic_fitting(self):
        self.cubic_fitting(self.roi_avg, self.baseline_correction.get())

    def open_setting(self):
        self.setting_window = tk.Toplevel(self.window)
        self.setting_window.title("Settings")

        # ROI setting
        tk.Label(self.setting_window, text="ROI:").grid(row=0, column=0)
        self.roi_var = tk.IntVar(value=self.ROI)
        self.roi_scale = tk.Scale(self.setting_window, from_=0, to_=900, orient=tk.HORIZONTAL, variable=self.roi_var,
                                  command=self.update_roi_display)
        self.roi_scale.grid(row=0, column=1)
        self.roi_value_label = tk.Label(self.setting_window, text=str(self.ROI))
        self.roi_value_label.grid(row=0, column=2)

        # Rows Number setting
        tk.Label(self.setting_window, text="Rows Number:").grid(row=1, column=0)
        self.rows_number_var = tk.IntVar(value=self.rows_number)
        self.rows_number_scale = tk.Scale(self.setting_window, from_=1, to_=30, orient=tk.HORIZONTAL,
                                          variable=self.rows_number_var, command=self.update_rows_number_display)
        self.rows_number_scale.grid(row=1, column=1)
        self.rows_number_value_label = tk.Label(self.setting_window, text=str(self.rows_number))
        self.rows_number_value_label.grid(row=1, column=2)

        # Brightness setting
        tk.Label(self.setting_window, text="Exposure (ms):").grid(row=2, column=0)
        self.brightness_var = tk.IntVar(value=self.brightness)
        self.brightness_scale = tk.Scale(self.setting_window, from_=0, to_=800, orient=tk.HORIZONTAL,
                                         variable=self.brightness_var,
                                         command=self.update_brightness_display)
        self.brightness_scale.grid(row=2, column=1)
        self.brightness_value_label = tk.Label(self.setting_window, text=str(self.brightness))
        self.brightness_value_label.grid(row=2, column=2)

        # Gain setting
        tk.Label(self.setting_window, text="Gain:").grid(row=3, column=0)
        self.gain_var = tk.IntVar(value=self.gain)
        self.gain_scale = tk.Scale(self.setting_window, from_=1, to_=32, orient=tk.HORIZONTAL, variable=self.gain_var,
                                   command=self.update_gain_display)
        self.gain_scale.grid(row=3, column=1)
        self.gain_value_label = tk.Label(self.setting_window, text=str(self.gain))
        self.gain_value_label.grid(row=3, column=2)

        # X-axis range setting
        tk.Label(self.setting_window, text="X-axis Min:").grid(row=4, column=0)
        self.x_axis_min_entry = tk.Entry(self.setting_window, textvariable=self.x_axis_min_var)
        self.x_axis_min_entry.grid(row=4, column=1)

        tk.Label(self.setting_window, text="X-axis Max:").grid(row=5, column=0)
        self.x_axis_max_entry = tk.Entry(self.setting_window, textvariable=self.x_axis_max_var)
        self.x_axis_max_entry.grid(row=5, column=1)

        self.x_axis_min_var.trace("w", self.update_x_axis_range)
        self.x_axis_max_var.trace("w", self.update_x_axis_range)

    def update_roi_display(self, value):
        self.ROI = int(value)
        self.roi_value_label.config(text=str(self.ROI))

    def update_rows_number_display(self, value):
        self.rows_number = int(value)
        self.rows_number_value_label.config(text=str(self.rows_number))

    def update_brightness_display(self, value):
        self.brightness = int(value)
        self.brightness_value_label.config(text=str(self.brightness))
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)  # Dynamically update camera brightness

    def update_gain_display(self, value):
        self.gain = int(value)
        self.gain_value_label.config(text=str(self.gain))
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)  # Dynamically update camera gain

    def update_x_axis_range(self, *args):
        try:
            x_min = self.safe_get_float(self.x_axis_min_var, 1)  # 默认为 1
            x_max = self.safe_get_float(self.x_axis_max_var, 1280)
            if x_min < x_max:
                self.ax.set_xlim([x_min, x_max])  # Set the X-axis limits based on user input
                self.canvas.draw()  # Update the plot with new X-axis range
        except ValueError:
            # Ignore if the inputs are not valid numbers
            pass

    def safe_get_float(self, var, default):
        try:
            value = float(var.get())
            return value
        except (ValueError, TclError):
            return default

    def save_data(self):
        # Prompt user to select save path and save data
        filepath = filedialog.asksaveasfilename(defaultextension=".txt")
        if filepath:
            np.savetxt(filepath, self.ax.lines[0].get_ydata(), fmt='%f')

    def exit_app(self):
        self.save_config()
        # Finalize device
        self.finalize_device()
        self.window.quit()
        self.window.destroy()

    def save_config(self):
        """将当前设置保存到 config.txt"""
        config = {
            "x_axis_min": self.x_axis_min_var.get(),
            "x_axis_max": self.x_axis_max_var.get(),
            "ROI": self.ROI,
            "rows_number": self.rows_number,
            "brightness": self.brightness,
            "gain": self.gain,
            "baseline_correction": self.baseline_correction.get()
        }
        try:
            with open("config.txt", "w") as file:
                json.dump(config, file)
            print("Config saved successfully")
        except Exception as e:
            print(f"Failed to save config: {e}")
# Create window and start application
app = App(tk.Tk(), "ROI Monitor")
