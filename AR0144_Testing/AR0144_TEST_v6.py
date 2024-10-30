import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.simpledialog as simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from itertools import combinations
from datetime import datetime
import ctypes
from ctypes import c_void_p, c_int, byref, create_string_buffer, wintypes
import json
import os
import logging
import math
import requests

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeviceController:
    def __init__(self):
        self.dll = ctypes.WinDLL('Dll/SpectroChipsControl.dll')
        self.initialize_functions()
        self.ROI = 470  # Default ROI value
        self.WL = [0, 0, 0, 0]  # Default wavelength coefficients
        self.x_axis_wavelength = None
        self.x_axis_pixel = list(range(1280))

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

        self.SP_DataWrite = self.dll.SP_DataWrite
        self.SP_DataWrite.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
        self.SP_DataWrite.restype = ctypes.c_long

    def initialize_device(self):
        # Initialize device
        hr = self.SP_Initialize(None)
        if hr != 0:  # ERROR_SUCCESS is 0
            logging.error("Device initialize fail")
            return False
        logging.info("Device initialized successfully")
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
            logging.info(f'Cleaned Data: {final_string}')

            # Parse JSON data
            try:
                json_data = json.loads(final_string)
                roi = json_data.get("ROI")
                if roi:
                    logging.info(f'ROI parameter: {roi}')
                    self.ROI = int(roi[2])  # Initial ROI value
                    self.WL = [float(x) for x in json_data.get("WL")]
                    self.x_axis_wavelength = [self.WL[0] + x * self.WL[1] + x * x * self.WL[2] + x * x * x * self.WL[3] for x in range(1280)]
                    self.x_axis_pixel = list(range(1280))
                else:
                    logging.warning("ROI parameter not found in JSON data. Using default settings.")
                    self.use_default_settings()
            except json.JSONDecodeError as e:
                logging.error(f'JSONDecodeError: {e}. Using default settings.')
                self.use_default_settings()
        else:
            logging.error(f'Read Failed with error code: {result}')
            self.use_default_settings()

    def write_input_data_to_flash(self, input_data):
        if not self.initialize_device():
            logging.error("Device initialization failed!")
            return False

        try:
            if len(input_data.strip()) == 0:
                logging.info("Abort (zero input)")
                return False

            # Convert the input to bytes
            input_data_bytes = input_data.encode('utf-8')  # Assuming wide character input

            # Write the input data to flash (assuming input length in bytes)
            data_length = len(input_data_bytes)
            input_buffer = ctypes.create_string_buffer(input_data_bytes, data_length)
            result = self.SP_DataWrite(input_buffer, data_length)

            # Check if write operation was successful
            if result == 0:
                logging.info("Write input data: Ok")
                logging.info(f'Input Data:{input_data}')
                return True
            else:
                logging.error("Write input data: Fail")
                return False

        finally:
            self.finalize_device()
    def use_default_settings(self):
        # Apply default settings
        self.ROI = 470
        self.WL = [176.1, 0.6378, 1.515e-5, 0]
        self.x_axis_wavelength = [self.WL[0]+ x * self.WL[1] + x * x * self.WL[2] for x in range(1280)]
        self.x_axis_pixel = list(range(1280))
        logging.info("Default settings applied due to an error.")

    def finalize_device(self):
        # Finalize device
        hr = self.SP_Finalize(None)
        if hr != 0:
            logging.error("Device finalize fail")
        else:
            logging.info("Device finalized successfully")


class CameraController:
    def __init__(self):
        self.cap = self.find_and_initialize_camera()
        if not self.cap:
            raise ValueError("No suitable camera found")
        else:
            logging.info("Camera initialized successfully")

    def find_and_initialize_camera(self):
        # Check for available cameras and their resolutions
        for i in range(10):  # Assuming a maximum of 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if width == 1280 and height == 800:
                    logging.info(f"Camera {i} initialized with resolution 1280x800")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
                    cap.set(cv2.CAP_PROP_MODE, 2)
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    return cap
                cap.release()
        return None

    def release_camera(self):
        if self.cap:
            self.cap.release()


class DataProcessor:
    def __init__(self):
        self.standard_wavelength = np.array([365.34, 406.15, 436.00, 545.79, 578.60, 696.56, 706.58, 727.17,
                                             738.34, 750.66, 763.56, 772.34, 794.56, 800.98, 811.48, 826.63,
                                             842.33, 852.20, 866.79, 912.38, 922.18])

    def calculate_fwhm(self, spectrum, width, prominence, height, baseline_correction):
        try:
            height_value = float(height)
        except ValueError:
            height_value = None

        if baseline_correction:
            spectrum = spectrum - np.average(spectrum[0:100])

        peaks, _ = find_peaks(spectrum[0:1260], width=width, prominence=prominence, height=height_value, distance=6)
        fwhm_values = []
        for peak in peaks:
            peak_height = spectrum[peak]
            half_height = peak_height / 2.0
            left_idx = peak - np.argmax(spectrum[peak::-1] <= half_height)
            right_idx = peak + np.argmax(spectrum[peak:] <= half_height)
            fwhm = round((right_idx - left_idx), 1)
            fwhm_values.append(fwhm)

        return peaks, fwhm_values

    def cubic_fitting(self, peaks):
        if len(peaks) < 3:
            messagebox.showwarning("Warning", "峰值數量不足，無法進行擬合")
            return None, None, None, None  # 返回四個 None

        # 定義擬合函數（多項式函數）
        def poly_func(x, a, b, c):
            return a * x ** 2 + b * x + c

        # 初始擬合，假設峰值與標準波長按順序匹配
        matched_wavelengths = self.standard_wavelength[:len(peaks)]
        try:
            popt, _ = curve_fit(poly_func, peaks, matched_wavelengths)
            fitted_peaks = poly_func(peaks, *popt)
            r_squared = np.corrcoef(matched_wavelengths, fitted_peaks)[0, 1] ** 2
        except Exception as e:
            messagebox.showerror("Error", f"初始擬合失敗: {e}")
            return None, None, None, None  # 返回四個 None

        # 如果擬合優度低於0.9999，嘗試所有可能的標準波長組合
        if r_squared < 0.9999:
            best_r2 = r_squared
            best_popt = popt
            best_fitted_peaks = fitted_peaks
            best_matched_wavelengths = matched_wavelengths

            num_combinations = 0  # 計算已嘗試的組合數
            max_combinations = 100000  # 設置最大嘗試組合數，防止計算時間過長

            # 遍歷所有可能的標準波長組合
            for comb in combinations(self.standard_wavelength, len(peaks)):
                selected_wavelengths = np.array(comb)
                num_combinations += 1

                if num_combinations > max_combinations:
                    messagebox.showwarning("Warning", "嘗試組合數過多，未能找到滿足條件的擬合")
                    break

                try:
                    popt_tmp, _ = curve_fit(poly_func, peaks, selected_wavelengths)
                    fitted_peaks_tmp = poly_func(peaks, *popt_tmp)
                    r_squared_tmp = np.corrcoef(selected_wavelengths, fitted_peaks_tmp)[0, 1] ** 2

                    if r_squared_tmp > best_r2:
                        best_r2 = r_squared_tmp
                        best_popt = popt_tmp
                        best_fitted_peaks = fitted_peaks_tmp
                        best_matched_wavelengths = selected_wavelengths

                        if best_r2 >= 0.9999:
                            break  # 已經達到目標，退出循環
                except Exception:
                    continue  # 如果擬合失敗，跳過這個組合

                if num_combinations % 10000 == 0:
                    print(f"已處理 {num_combinations} 個組合，目前最佳 R²: {best_r2:.6f}")

            if best_r2 < 0.9999:
                messagebox.showwarning("Warning", "無法達到 R² ≥ 0.9999 的擬合")
            return best_popt, best_r2, best_fitted_peaks, best_matched_wavelengths
        else:
            return popt, r_squared, fitted_peaks, matched_wavelengths


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.rows_number = 10  # Initial number of rows
        self.brightness = 75  # Initial Brightness value
        self.gain = 2  # Initial Gain value
        self.baseline_correction = tk.BooleanVar(value=True)  # Initial baseline correction value

        self.width = 4
        self.prominence = 7.0
        self.height = 250

        self.x_axis_min_var = tk.DoubleVar(value=0)  # Default min value of X-axis
        self.x_axis_max_var = tk.DoubleVar(value=1280)  # Default max value of X-axis

        self.autoROI_var = tk.BooleanVar(value=False)  # Initial Auto ROI value

        self.load_config()

        # Initialize the device and camera
        self.device_controller = DeviceController()
        if self.device_controller.initialize_device():
            self.device_controller.read_data_from_device()
        else:
            raise ValueError("No suitable device found")

        self.camera_controller = CameraController()
        self.cap = self.camera_controller.cap

        # Data processor
        self.data_processor = DataProcessor()

        # Initialize variables before update()
        self.roi_var = tk.IntVar(value=self.device_controller.ROI)

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
                    self.rows_number = config.get("rows_number", 10)
                    self.brightness = config.get("brightness", 75)
                    self.gain = config.get("gain", 2)
                    self.baseline_correction.set(config.get("baseline_correction", True))
                    self.autoROI_var.set(config.get("autoROI", False))

                    # 添加读取 width、prominence、height 的代码
                    self.width = config.get("width", 4)
                    self.prominence = config.get("prominence", 7.0)
                    self.height = config.get("height", 250)

                    logging.info("Config loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
                messagebox.showerror("Error", f"Failed to load config: {e}")
        else:
            # 设置默认值
            self.width = 4
            self.prominence = 7.0
            self.height = 250
            logging.info("Config file not found, using default settings")
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
            combined = self.process_frame(frame)
            result = combined.reshape(800, 1280).astype(np.float64)
            normalized_result = (result / 4096 * 254).astype(np.uint8)

            if self.autoROI_var.get():
                # Calculate the average of each row
                row_averages = np.mean(result, axis=1)
                # Find the index of the row with the highest average value
                max_row_index = np.argmax(row_averages)
                self.device_controller.ROI = max_row_index
                # Update ROI label if settings window is open
                if hasattr(self, 'roi_value_label'):
                    self.roi_value_label.config(text=str(self.device_controller.ROI))
                # Also update the roi_var if it exists
                if hasattr(self, 'roi_var'):
                    self.roi_var.set(self.device_controller.ROI)
            else:
                # Ensure that the ROI is set from user input
                self.device_controller.ROI = self.roi_var.get()

            start_point = (0, self.device_controller.ROI - self.rows_number)  # Start point coordinates
            end_point = (normalized_result.shape[1], self.device_controller.ROI + self.rows_number)  # End point coordinates
            color = (255, 0, 0)  # BGR color value, red
            thickness = 2  # Line thickness

            # Draw red rectangle on normalized_result
            cv2.rectangle(normalized_result, start_point, end_point, color, thickness)
            resized_image = cv2.resize(normalized_result, (0, 0), fx=0.4, fy=0.5)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            self.camera_frame.config(image=self.photo)
            self.analyze_roi(result, self.device_controller.ROI, self.rows_number)
        self.window.after(self.delay, self.update)

    def process_frame(self, frame):
        # 將高位和低位資料提取並組合
        high_bits = frame[0][::2].astype(np.uint16)
        low_bits = frame[0][1::2].astype(np.uint16)
        if len(frame[0]) % 2 != 0:
            low_bits = np.append(low_bits, 0)
        combined_data = ((high_bits << 4) | (low_bits - 128)).astype(np.uint16)
        return combined_data

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

        x_axis = self.device_controller.x_axis_wavelength if self.x_axis_option.get() == "Wavelength" else self.device_controller.x_axis_pixel
        self.ax.plot(x_axis, data)

        x_min = self.safe_get_float(self.x_axis_min_var, 0)
        x_max = self.safe_get_float(self.x_axis_max_var, 1280)
        self.ax.set_xlim([x_min, x_max])  # Apply the new X-axis limits

        self.ax.set_ylim([0, 4100])  # Set y-axis limits
        self.ax.set_title("Real-time Spectrum")
        xlabel = "Wavelength (nm)" if self.x_axis_option.get() == "Wavelength" else "Pixel"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Intensity (a.u.)")

        self.canvas.draw()

    def update_plot_x_axis(self, event=None):
        self.draw_plot(self.roi_avg)

    def perform_cubic_fitting(self):
        if not hasattr(self, 'roi_avg'):
            messagebox.showwarning("Warning", "No data available for fitting.")
            return

        roi_avg = self.roi_avg.copy()
        if self.baseline_correction.get():
            roi_avg = roi_avg - np.average(roi_avg[0:100])

        data_processor = self.data_processor
        data_processor = self.data_processor

        peaks, fwhm_values = data_processor.calculate_fwhm(
            roi_avg,
            width=self.width,
            prominence=self.prominence,
            height=self.height,
            baseline_correction=False
        )
        result = data_processor.cubic_fitting(peaks)

        if result is None or any(v is None for v in result):
            messagebox.showwarning("Warning", "擬合失敗，無法進行後續操作。")
            return

        popt, r_squared, fitted_peaks, matched_wavelengths = result

        result_text = f"Fitting Curve Equation: {popt[0]:.5e}x² + {popt[1]:.5e}x + {popt[2]:.5e}\nR²: {r_squared:.6f}"

        # Create a new window to display results
        result_window = tk.Toplevel()
        result_window.title("Cubic Fitting Results")

        fig_fit, ax_fit = plt.subplots()
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=result_window)
        canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ax_fit.cla()
        ax_fit.plot(peaks, matched_wavelengths, 'ro', label='Standard Wavelengths', ms=3)
        if popt is not None:
            ax_fit.plot(peaks, fitted_peaks, 'b--', label='Fitted Curve')
        ax_fit.set_xlabel('Peak Position (pixel)')
        ax_fit.set_ylabel('Standard Wavelength (nm)')
        ax_fit.legend()
        canvas_fit.draw()

        # Display the fitting results in the new window
        fitting_label = tk.Label(result_window, text=result_text)
        fitting_label.pack()

        # Create and update the FWHM table
        fwhm_table = self.create_table(result_window)

        # Convert FWHM from pixels to nm using the derivative of the fitting function
        def poly_derivative(x):
            return 2 * popt[0] * x + popt[1]

        FWHM_convert = [poly_derivative(peak) * fwhm for peak, fwhm in zip(peaks, fwhm_values)]
        self.update_table_with_fwhm_nm(fwhm_table, peaks, fitted_peaks, FWHM_convert)

    def create_table(self, parent):
        tree = ttk.Treeview(parent, columns=('Peak Position', 'Wavelength', 'FWHM(nm)'), show='headings')
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.heading('Peak Position', text='Peak Position')
        tree.heading('Wavelength', text='Wavelength\n(nm)')
        tree.heading('FWHM(nm)', text='FWHM\n(nm)')

        tree.column('Peak Position', anchor='center', width=80)
        tree.column('Wavelength', anchor='center', width=80)
        tree.column('FWHM(nm)', anchor='center', width=80)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        button_frame = tk.Frame(parent)
        button_frame.pack(side='bottom', fill='x')

        write_json_button = ttk.Button(button_frame, text='Write JSON', command=self.write_json)
        write_json_button.pack(side='left')

        return tree

    def write_json(self):
        # Open dialogs to get CHIP_ID and S/N
        chip_id = simpledialog.askstring("Input", "Please input CHIP_ID:")
        if chip_id is None or len(chip_id.strip()) == 0:
            messagebox.showinfo("Info", "CHIP_ID is required. Operation aborted.")
            return

        serial_number = simpledialog.askstring("Input", "Please input S/N:")
        if serial_number is None or len(serial_number.strip()) == 0:
            messagebox.showinfo("Info", "S/N is required. Operation aborted.")
            return

        # Collect other information
        current_date = datetime.now().strftime("%Y-%m-%d")  # Get current date as a string
        roi = self.device_controller.ROI
        wl = self.device_controller.WL
        exp = self.brightness  # Exposure time
        gain = self.gain       # Gain

        # Create a dictionary to hold all information
        data_dict = {
            "CHIP_ID": chip_id,
            "S/N": serial_number,
            "Date": current_date,
            "ROI": [0,1280,roi, 20],
            "WL": wl,
            "EXP": exp,
            "Gain": gain
        }

        # Convert the dictionary to a JSON string
        input_data = json.dumps(data_dict)

        # Call the method to write data to device
        success = self.device_controller.write_input_data_to_flash(input_data)
        if success:
            messagebox.showinfo("Success", "Write input data: Ok")
        else:
            messagebox.showerror("Error", "Write input data: Fail")

    def update_table_with_fwhm_nm(self, table, peaks, wavelengths, FWHM_convert):
        for i in table.get_children():
            table.delete(i)

        for peak, wavelength, fwhm_nm in zip(peaks, wavelengths, FWHM_convert):
            if fwhm_nm is not None and not math.isnan(fwhm_nm) and fwhm_nm <= 9.5:
                table.insert('', 'end', values=(peak, f"{wavelength:.2f}", f"{fwhm_nm:.2f}"))

    def open_setting(self):
        self.setting_window = tk.Toplevel(self.window)
        self.setting_window.title("Settings")

        # ROI setting
        tk.Label(self.setting_window, text="ROI:").grid(row=0, column=0)
        self.roi_scale = tk.Scale(self.setting_window, from_=0, to_=900, orient=tk.HORIZONTAL, variable=self.roi_var)
        self.roi_scale.grid(row=0, column=1)
        self.roi_value_label = tk.Label(self.setting_window, text=str(self.device_controller.ROI))
        self.roi_value_label.grid(row=0, column=2)

        # Auto ROI setting
        self.autoROI_checkbox = tk.Checkbutton(self.setting_window, text="Auto ROI", variable=self.autoROI_var)
        self.autoROI_checkbox.grid(row=0, column=3)

        # Rows Number setting
        tk.Label(self.setting_window, text="Rows Number:").grid(row=1, column=0)
        self.rows_number_var = tk.IntVar(value=self.rows_number)
        self.rows_number_scale = tk.Scale(self.setting_window, from_=1, to_=30, orient=tk.HORIZONTAL,
                                          variable=self.rows_number_var)
        self.rows_number_scale.grid(row=1, column=1)
        self.rows_number_value_label = tk.Label(self.setting_window, text=str(self.rows_number))
        self.rows_number_value_label.grid(row=1, column=2)

        # Brightness setting
        tk.Label(self.setting_window, text="Exposure (ms):").grid(row=2, column=0)
        self.brightness_var = tk.IntVar(value=self.brightness)
        self.brightness_scale = tk.Scale(self.setting_window, from_=0, to_=800, orient=tk.HORIZONTAL,
                                         variable=self.brightness_var)
        self.brightness_scale.grid(row=2, column=1)
        self.brightness_value_label = tk.Label(self.setting_window, text=str(self.brightness))
        self.brightness_value_label.grid(row=2, column=2)

        # Gain setting
        tk.Label(self.setting_window, text="Gain:").grid(row=3, column=0)
        self.gain_var = tk.IntVar(value=self.gain)
        self.gain_scale = tk.Scale(self.setting_window, from_=1, to_=32, orient=tk.HORIZONTAL, variable=self.gain_var)
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

        # Confirm and Cancel buttons
        button_frame = tk.Frame(self.setting_window)
        button_frame.grid(row=6, column=0, columnspan=4, pady=10)
        save_button = ttk.Button(button_frame, text="Save", command=self.save_settings)
        save_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.setting_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

        # Bind variable changes to update labels and plot
        self.roi_var.trace("w", self.update_settings)
        self.rows_number_var.trace("w", self.update_settings)
        self.brightness_var.trace("w", self.update_settings)
        self.gain_var.trace("w", self.update_settings)
        self.x_axis_min_var.trace("w", self.update_x_axis_range)
        self.x_axis_max_var.trace("w", self.update_x_axis_range)
        self.autoROI_var.trace("w", self.update_settings)

    def update_settings(self, *args):
        # Update labels
        self.roi_value_label.config(text=str(self.roi_var.get()))
        self.rows_number_value_label.config(text=str(self.rows_number_var.get()))
        self.brightness_value_label.config(text=str(self.brightness_var.get()))
        self.gain_value_label.config(text=str(self.gain_var.get()))

        # Update settings in real-time
        self.rows_number = self.rows_number_var.get()
        self.brightness = self.brightness_var.get()
        self.gain = self.gain_var.get()

        # Update camera settings
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)

        if self.autoROI_var.get():
            # Disable the ROI Scale
            self.roi_scale.config(state='disabled')
        else:
            # Enable the ROI Scale
            self.roi_scale.config(state='normal')
            self.device_controller.ROI = self.roi_var.get()

        # Redraw the plot
        if hasattr(self, 'roi_avg'):
            self.draw_plot(self.roi_avg)

    def update_x_axis_range(self, *args):
        try:
            x_min = self.safe_get_float(self.x_axis_min_var, 1)  # 默認為 1
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
        except (ValueError, tk.TclError):
            return default

    def save_settings(self):
        # Save settings and close the window
        self.setting_window.destroy()

    def save_data(self):
        # 提示用户选择保存路径并保存数据
        filepath = filedialog.asksaveasfilename(defaultextension=".txt")
        if filepath:
            y_data = self.ax.lines[0].get_ydata()
            if self.x_axis_option.get() == "Wavelength":
                x_data = self.device_controller.x_axis_wavelength
                header = 'Wavelength\tIntensity'
            else:
                x_data = self.device_controller.x_axis_pixel
                header = 'Pixel\tIntensity'
            data_to_save = np.column_stack((x_data, y_data))
            np.savetxt(filepath, data_to_save, fmt='%f', header=header, delimiter='\t', comments='')

    def exit_app(self):
        self.save_config()
        # Finalize device
        self.device_controller.finalize_device()
        # Release camera resources
        self.camera_controller.release_camera()
        self.window.quit()
        self.window.destroy()

    def save_config(self):
        """將當前設置保存到 config.txt"""
        config = {
            "x_axis_min": self.x_axis_min_var.get(),
            "x_axis_max": self.x_axis_max_var.get(),
            "rows_number": self.rows_number,
            "brightness": self.brightness,
            "gain": self.gain,
            "baseline_correction": self.baseline_correction.get(),
            "autoROI": self.autoROI_var.get(),
            "width": self.width,
            "prominence": self.prominence,
            "height": self.height
        }
        try:
            with open("config.txt", "w") as file:
                json.dump(config, file)
            logging.info("Config saved successfully")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            messagebox.showerror("Error", f"Failed to save config: {e}")

# Create window and start application
if __name__ == "__main__":
    app = App(tk.Tk(), "ROI Monitor")
