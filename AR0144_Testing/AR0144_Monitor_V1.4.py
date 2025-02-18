import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
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
from ctypes import c_void_p, c_int, byref, wintypes
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
        self.gain = 2
        self.EXP = 450


    def initialize_functions(self):
        # Define function prototypes from DLL
        self.SP_Initialize = self.dll.SP_Initialize
        self.SP_Initialize.argtypes = [c_void_p]
        self.SP_Initialize.restype = wintypes.DWORD

        self.SP_Finalize = self.dll.SP_Finalize
        self.SP_Finalize.argtypes = [c_void_p]
        self.SP_Finalize.restype = wintypes.DWORD

        self.SP_DataRead = self.dll.SP_DataRead
        self.SP_DataRead.argtypes = [c_void_p, ctypes.POINTER(c_int)]
        self.SP_DataRead.restype = ctypes.c_long

        self.SP_DataWrite = self.dll.SP_DataWrite
        self.SP_DataWrite.argtypes = [ctypes.POINTER(ctypes.c_char), c_int]
        self.SP_DataWrite.restype = ctypes.c_long

        self.SP_RegisterRead = self.dll.SP_RegisterRead
        self.SP_RegisterRead.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_ubyte)]
        self.SP_RegisterRead.restype = ctypes.c_ulong

        self.SP_GetVersion = self.dll.SP_GetVersion
        self.SP_GetVersion.argtypes = [ctypes.POINTER(ctypes.c_ushort)]
        self.SP_GetVersion.restype = ctypes.c_ulong

    def initialize_device(self):
        # Initialize the spectrochip device
        hr = self.SP_Initialize(None)
        if hr != 0:  # ERROR_SUCCESS is 0
            logging.error("Device initialize fail")
            return False
        logging.info("Device initialized successfully")
        return True

    def finalize_device(self):
        # Properly finalize device to free resources
        hr = self.SP_Finalize(None)
        if hr != 0:
            logging.error("Device finalize fail")
        else:
            logging.info("Device finalized successfully")

    def read_data_from_device(self):
        # Allocate buffer
        buffer_size = 4096  # 4KB
        buffer = (ctypes.c_ubyte * buffer_size)()
        data_length = c_int(buffer_size)
        self.SP_Initialize(None)
        # Call the SP_DataRead function
        result = self.SP_DataRead(buffer, byref(data_length))

        # Check the result and process
        if result == 0:  # Assuming 0 indicates success
            read_data = bytes(buffer[:data_length.value]).decode('utf-8')
            cleaned_data = re.sub(r'[\x00\r]', '', read_data)
            logging.info(f'Cleaned Data: {cleaned_data}')

            # Parse JSON data
            try:
                json_data = json.loads(cleaned_data)
                self.apply_device_settings(json_data)
            except json.JSONDecodeError as e:
                logging.warning(f'JSONDecodeError: {e}. Using default settings.')

        else:
            logging.error(f'Read Failed with error code: {result}')

        # Get FW version
        version = ctypes.c_ushort(0)
        hr = self.SP_GetVersion(ctypes.byref(version))
        if hr == 0:
            logging.info(f"FW version: 0x{version.value:04X}")

        # Check USB condition
        address = '0x615'
        ul_address = int(address, 16)
        value = ctypes.c_ubyte(0)
        hr = self.SP_RegisterRead(ul_address, ctypes.byref(value))

        if hr == 0:
            # Determine USB environment
            if value.value == 2:
                logging.info("Currently connected to USB 3.0 environment")
            elif value.value == 1:
                logging.warning(
                    "Currently connected to USB 2.0 environment, unexpected bugs may occur. It is recommended to switch to a USB 3.0 environment")

    def apply_device_settings(self, json_data):
        roi = json_data.get("roi_height")
        if roi:
            logging.info(f'ROI parameter: {roi}')
            self.ROI =   int(roi)  # Initial ROI value
            self.WL = [float(json_data.get("conversion_factor_0_a0")), float(json_data.get("conversion_factor_0_a1")), float(json_data.get("conversion_factor_0_a2")), float(json_data.get("conversion_factor_0_a3"))]
            self.x_axis_wavelength = [self.WL[0] + x * self.WL[1] + x * x * self.WL[2] + x * x * x * self.WL[3] for x in range(1280)]
            self.x_axis_pixel = list(range(1280))
            self.EXP = self.parse_int(json_data.get("init_exp_value"), "EXP")
            self.gain = self.parse_int(json_data.get("analog_gain"), "gain")
        else:
            logging.warning("ROI parameter not found in JSON data. Using default settings.")

    def parse_int(self, value, name):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            logging.warning(f"Warning: Issue setting {name}: {e}")
            return getattr(self, name)

    def write_input_data_to_flash(self, input_data):
        if not self.initialize_device():
            logging.error("Device initialization failed!")
            return False

        try:
            if not input_data.strip():
                logging.info("Abort (zero input)")
                return False

            input_data_bytes = input_data.encode('utf-8')
            data_length = len(input_data_bytes)
            input_buffer = ctypes.create_string_buffer(input_data_bytes, data_length)
            result = self.SP_DataWrite(input_buffer, data_length)

            if result == 0:
                logging.info("Write input data: Ok")
                logging.info(f'Input Data: {input_data}')
                return True
            else:
                logging.error("Write input data: Fail")
                return False

        finally:
            self.finalize_device()

class CameraController:
    def __init__(self):
        self.cap = self.find_and_initialize_camera()
        if not self.cap:
            raise ValueError("No suitable camera found")
        else:
            logging.info("Camera initialized successfully")

    def find_and_initialize_camera(self):
        # Check for available cameras and initialize one with the required resolution
        for i in range(10):   # Iterate over possible camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # Select camera only if it matches the resolution criteria
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
        with open("Dll/config.txt", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.standard_wavelength = np.array(config['standard_wavelength'])
        self.font_size = 14
        #self.window.bind("<Control-MouseWheel>", self.on_mouse_wheel)

    def calculate_fwhm(self, spectrum, width, prominence, height, baseline_correction, shift):
        try:
            height_value = float(height)
        except ValueError:
            height_value = None

        if baseline_correction:
            # Apply baseline correction by subtracting the average of the first 100 data points
            spectrum = spectrum - np.average(spectrum[:100])

        # Find peaks within the given constraints
        peaks, _ = find_peaks(spectrum[:1260], width=width, prominence=prominence, height=height_value, distance=6)

        fwhm_values = []
        refined_peaks = []

        for peak in peaks:
            # Get left and right points based on shift
            left_idx = max(peak - shift, 0)
            right_idx = min(peak + shift, len(spectrum) - 1)

            # Extract the three points
            x_data = np.array([left_idx, peak, right_idx])  # Ensure x_data is a NumPy array
            y_data = np.array(spectrum[x_data])  # Ensure y_data is also a NumPy array

            # Perform quadratic fitting
            try:
                coeffs, _ = curve_fit(lambda x, a, b, c: a * x ** 2 + b * x + c, x_data, y_data)
                a, b, c = coeffs

                # Validate the quadratic function
                if a < 0:  # Ensure the parabola opens downward
                    # Calculate the peak position and FWHM from the quadratic fit
                    peak_position = -b / (2 * a)
                    peak_height = a * peak_position ** 2 + b * peak_position + c
                    half_height = peak_height / 2

                    # Solve for the left and right intersection points at half height
                    delta = b ** 2 - 4 * a * (c - half_height)
                    if delta >= 0:

                        # Find left intersection point
                        left_idx = peak
                        while left_idx > 0 and spectrum[left_idx] > half_height:
                            left_idx -= 1

                        # Interpolate for precise left half-height position
                        if left_idx > 0:
                            left_x = left_idx + (half_height - spectrum[left_idx]) / (
                                        spectrum[left_idx + 1] - spectrum[left_idx])
                        else:
                            left_x = left_idx

                        # Find right intersection point
                        right_idx = peak
                        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_height:
                            right_idx += 1

                        # Interpolate for precise right half-height position
                        if right_idx < len(spectrum) - 1:
                            right_x = right_idx - (spectrum[right_idx] - half_height) / (
                                        spectrum[right_idx] - spectrum[right_idx - 1])
                        else:
                            right_x = right_idx

                        # Calculate FWHM
                        fwhm = round(abs(right_x - left_x), 2)
                        fwhm_values.append(fwhm)
                        refined_peaks.append(peak_position)

            except Exception as e:
                print(f"Error fitting peak at index {peak}: {e}")

        return refined_peaks, fwhm_values

    def calculate_fwhm_with_plot(self, spectrum, width, prominence, height, baseline_correction, shift):
        try:
            height_value = float(height)
        except ValueError:
            height_value = None

        if baseline_correction:
            # Apply baseline correction by subtracting the average of the first 100 data points
            spectrum = spectrum - np.average(spectrum[:100])

        # Find peaks within the given constraints
        peaks, _ = find_peaks(spectrum[:1240], width=width, prominence=prominence, height=height_value, distance=16) #6

        fwhm_values = []
        refined_peaks = []
        fit_results = []  # Store fit results for plotting
        fwhm_annotations = []  # Store FWHM annotations for plotting

        for peak in peaks:
            # Get left and right points based on shift
            left_idx = max(peak - shift, 0)
            right_idx = min(peak + shift, len(spectrum) - 1)

            # Extract the three points
            x_data = np.array([left_idx, peak, right_idx])  # Ensure x_data is a NumPy array
            y_data = np.array(spectrum[x_data])  # Ensure y_data is also a NumPy array

            # Perform quadratic fitting
            try:
                coeffs, _ = curve_fit(lambda x, a, b, c: a * x ** 2 + b * x + c, x_data, y_data)
                a, b, c = coeffs

                # Validate the quadratic function
                if a < 0:  # Ensure the parabola opens downward
                    # Generate the fit curve
                    x_fit = np.linspace(left_idx, right_idx, 100)
                    y_fit = a * x_fit ** 2 + b * x_fit + c
                    fit_results.append((x_fit, y_fit))  # Store for plotting

                    # Calculate the peak position and FWHM from the quadratic fit
                    peak_position = -b / (2 * a)
                    peak_height = a * peak_position ** 2 + b * peak_position + c
                    half_height = peak_height / 2

                    # Solve for the left and right intersection points at half height
                    delta = b ** 2 - 4 * a * (c - half_height)
                    if delta >= 0:
                        # Find left intersection point
                        left_idx = peak
                        while left_idx > 0 and spectrum[left_idx] > half_height:
                            left_idx -= 1

                        # Interpolate for precise left half-height position
                        if left_idx > 0:
                            left_x = left_idx + (half_height - spectrum[left_idx]) / (
                                        spectrum[left_idx + 1] - spectrum[left_idx])
                        else:
                            left_x = left_idx

                        # Find right intersection point
                        right_idx = peak
                        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_height:
                            right_idx += 1

                        # Interpolate for precise right half-height position
                        if right_idx < len(spectrum) - 1:
                            right_x = right_idx - (spectrum[right_idx] - half_height) / (
                                        spectrum[right_idx] - spectrum[right_idx - 1])
                        else:
                            right_x = right_idx

                        # Calculate FWHM
                        fwhm = round(abs(right_x - left_x), 2)
                        fwhm_values.append(fwhm)
                        refined_peaks.append(round(peak_position, 3))

                        # Add FWHM annotation for plotting
                        fwhm_annotations.append((left_x, right_x, half_height))
            except Exception as e:
                print(f"Error fitting peak at index {peak}: {e}")
        return refined_peaks, fwhm_values

    @staticmethod
    def calculate_peak_fwhm(spectrum, peak):
        peak_height = spectrum[peak]
        half_height = peak_height / 2.0
        left_idx = peak - np.argmax(spectrum[peak::-1] <= half_height)
        right_idx = peak + np.argmax(spectrum[peak:] <= half_height)
        fwhm = round((right_idx - left_idx), 1)
        return fwhm

    def cubic_fitting(self, peaks):
        if len(peaks) < 5:
            messagebox.showwarning("Warning", "峰值數量不足，無法進行擬合")
            return None, None, None, None

        # 定義擬合函數（多項式函數）
        def poly_func(x, a, b, c):
            return a * x ** 2 + b * x + c

        # 初始擬合，假設峰值與標準波長按順序匹配
        matched_wavelengths = self.standard_wavelength[:len(peaks)]
        popt, r_squared, fitted_peaks = self.try_curve_fit(peaks, matched_wavelengths, poly_func)

        # 如果擬合優度低於0.9999，嘗試所有可能的標準波長組合
        if r_squared < 0.9999:
            popt, r_squared, fitted_peaks, matched_wavelengths = self.try_all_combinations(peaks, poly_func, r_squared,
                                                                                           popt, fitted_peaks,
                                                                                           matched_wavelengths)

        return popt, r_squared, fitted_peaks, matched_wavelengths

    def try_curve_fit(self, peaks, matched_wavelengths, poly_func):
        try:
            # 確保數據為 numpy array
            peaks = np.array(peaks, dtype=np.float64)
            matched_wavelengths = np.array(matched_wavelengths, dtype=np.float64)

            # 確保數據形狀一致
            if peaks.shape != matched_wavelengths.shape:
                raise ValueError("peaks and matched_wavelengths must have the same shape")

            # 擬合曲線
            popt, _ = curve_fit(poly_func, peaks, matched_wavelengths)
            fitted_peaks = poly_func(peaks, *popt)

            # 計算 R²
            fitted_peaks = np.array(fitted_peaks, dtype=np.float64)  # 確保數據類型一致
            r_squared = np.corrcoef(matched_wavelengths, fitted_peaks)[0, 1] ** 2
        except Exception as e:
            messagebox.showerror("Error", f"初始擬合失敗: {e}")
            return None, None, None

        return popt, r_squared, fitted_peaks

    def try_all_combinations(
            self,
            peaks,
            poly_func,
            best_r2,
            best_popt,
            best_fitted_peaks,
            best_matched_wavelengths,
            threshold_peaks=750,
            threshold_std=650,
            max_combinations=150000
    ):

        # 1) Divide into groups
        peaks = np.array(peaks, dtype=np.float64)
        peaks_lower = peaks[peaks <= threshold_peaks]
        peaks_upper = peaks[peaks > threshold_peaks]

        std_wavelength = np.array(self.standard_wavelength, dtype=np.float64)
        std_lower = std_wavelength[std_wavelength <= threshold_std]
        std_upper = std_wavelength[std_wavelength > threshold_std]

        best_r2_upper = best_r2
        best_popt_upper = best_popt
        best_fitted_peaks_upper = best_fitted_peaks
        best_matched_wavelengths_upper = best_matched_wavelengths

        if len(peaks_upper) > 0 and len(std_upper) >= len(peaks_upper):
            logging.info(f"Starting search in upper range: peaks_upper={len(peaks_upper)}, std_upper={len(std_upper)}")
            best_r2_upper, best_popt_upper, best_fitted_peaks_upper, best_matched_wavelengths_upper = \
                self._search_combinations_for_subset(
                    subset_peaks=peaks_upper,
                    subset_std=std_upper,
                    poly_func=poly_func,
                    best_r2=best_r2_upper,
                    best_popt=best_popt_upper,
                    best_fitted_peaks=best_fitted_peaks_upper,
                    best_matched_wavelengths=best_matched_wavelengths_upper,
                    max_combinations=max_combinations
                )
        else:
            logging.info("Insufficient data in upper range, skipping search")

        # ======================================================
        # (C) Combine results from both groups and perform a single "global polynomial fit"
        # ======================================================
        final_r2 = best_r2
        best_popt_final = best_popt
        final_fitted_peaks = best_fitted_peaks
        final_matched_wavelengths_upper = best_matched_wavelengths

        # 1. Combine peaks
        final_peaks = np.concatenate([peaks_lower, peaks_upper])

        # 2. Combine matched wavelengths
        final_matched_wavelengths = np.concatenate([std_lower, best_matched_wavelengths_upper])

        # 3. Perform final curve fitting
        # final_popt, _ = curve_fit(poly_func, final_peaks, final_matched_wavelengths)
        final_r2, best_popt_final, final_fitted_peaks, final_matched_wavelengths_upper = \
            self._search_combinations_for_subset(
                subset_peaks=final_peaks,
                subset_std=final_matched_wavelengths,
                poly_func=poly_func,
                best_r2=final_r2,
                best_popt=best_popt_final,
                best_fitted_peaks=final_fitted_peaks,
                best_matched_wavelengths=final_matched_wavelengths_upper,
                max_combinations=max_combinations
            )

        logging.info(f"Final combined R² = {final_r2:.6f}")

        # ================ Return final results =================
        return best_popt_final, final_r2, final_fitted_peaks, final_matched_wavelengths_upper

    def _search_combinations_for_subset(
            self,
            subset_peaks,
            subset_std,
            poly_func,
            best_r2,
            best_popt,
            best_fitted_peaks,
            best_matched_wavelengths,
            max_combinations,
            r2_break=0.999999
    ):
        """
        Sub-function to search combinations for a specific subset (subset_peaks, subset_std).
        Displays the total number of possible combinations at the start and the number of attempted combinations at the end.
        Returns the updated (best_r2, best_popt, best_fitted_peaks, best_matched_wavelengths).
        """
        num_combinations = 0
        m = len(subset_peaks)

        # Calculate total possible combinations
        total_combinations = math.comb(len(subset_std), m) if len(subset_std) >= m else 0
        logging.info(f"Total possible combinations: {total_combinations}")

        for comb_ in combinations(subset_std, m):
            selected_wavelengths = np.array(comb_, dtype=np.float64)
            num_combinations += 1

            # Limit the number of attempted combinations
            if num_combinations > max_combinations:
                messagebox.showwarning("Warning", "Too many combinations attempted, unable to find a satisfactory fit")
                break

            try:
                # Perform fitting
                popt_tmp, _ = curve_fit(poly_func, subset_peaks, selected_wavelengths)
                fitted_peaks_tmp = poly_func(subset_peaks, *popt_tmp)

                # Calculate R²
                r_squared_tmp = np.corrcoef(selected_wavelengths, fitted_peaks_tmp)[0, 1] ** 2

                if r_squared_tmp > best_r2:
                    best_r2 = r_squared_tmp
                    best_popt = popt_tmp
                    best_fitted_peaks = fitted_peaks_tmp
                    best_matched_wavelengths = selected_wavelengths

                    # If the target R² is exceeded, terminate early
                    if best_r2 >= r2_break:
                        break

            except Exception:
                # Skip if fitting fails
                continue

            # Periodically log progress (adjust interval as needed)
            if num_combinations % 2000 == 0:
                logging.info(
                    f"Processed {num_combinations}/{total_combinations} combinations, current best R²: {best_r2:.6f}")

        logging.info(
            f"Completed. Attempted {num_combinations} combinations out of {total_combinations} possible combinations.")
        logging.info(f"Subset {subset_peaks}")
        logging.info(f"best_matched_wavelengths {best_matched_wavelengths}")
        return best_r2, best_popt, best_fitted_peaks, best_matched_wavelengths

    def simple_baseline_evaluation(self, data):
        """
        Calculate the absolute difference between the average of the first 50 and the last 50 intensity values.

        Parameters:
            data (numpy.ndarray): A 2D array where the first column is wavelength and the second column is intensity.

        Returns:
            float: The absolute difference between the average of the first 50 and the last 50 intensity values.
        """

        # Compute the mean of the first 20 values
        first_20_mean = np.mean(data[:20])

        # Compute the mean of the last 20 values
        last_20_mean = np.mean(data[-20:])

        # Calculate and return the absolute difference
        absolute_difference = abs(first_20_mean - last_20_mean)
        return absolute_difference


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize default parameters
        self.initialize_parameters()
        self.load_config()

        # Initialize the device and camera
        self.initialize_device_and_camera()

        # Data processor
        self.data_processor = DataProcessor()

        # Initialize variables before update()
        self.roi_var = tk.IntVar(value=self.device_controller.ROI)

        # Create GUI layout
        self.create_widgets()

        self.font_size = 14
        self.window.bind("<Control-MouseWheel>", self.on_mouse_wheel)
        self.plot_scale_factor = 1.0

        # Timer
        self.delay = 50
        self.update()

        # Initialize private attribute for the table
        self._fwhm_table = None
        self.photo = None
        self.camera_image_saved = None
        self.camera_image_tmp = None


        self.window.mainloop()
    def initialize_parameters(self):
        self.rows_number = 10
        self.brightness = 75
        self.gain = 2
        self.baseline_correction = tk.BooleanVar(value=True)
        self.width = 2.5
        self.prominence = 5.0
        self.height = 250
        self.x_axis_min_var = tk.DoubleVar(value=0)
        self.x_axis_max_var = tk.DoubleVar(value=1280)
        self.autoROI_var = tk.BooleanVar(value=False)
        self.lam = 1000
        self.p = 0.001
        self.niter = 5
        self.window_length = 5
        self.polyorder = 3
        self.baseline_eva = 0
    def connect_camera(self):
        if not self.connected:
            try:
                self.camera_controller = CameraController()
                self.cap = self.camera_controller.cap
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
                self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
                self.connected = True
                self.connect_button.config(state='disabled')
                self.disconnect_button.config(state='normal')
                logging.info("Camera connected")

                # Initialize the device and camera after connecting
                if self.device_controller.initialize_device():
                    #self.device_controller.read_data_from_device()
                    self.brightness = self.device_controller.EXP
                    self.gain = self.device_controller.gain
            except ValueError as e:
                messagebox.showerror("Error", str(e))

    def on_mouse_wheel(self, event):
        """通过 Ctrl + 鼠标滚轮调整字体大小和线图缩放"""
        if event.delta > 0:  # 滚轮向上滚动
            self.font_size += 1
            self.plot_scale_factor *= 1.1  # 放大线图
        elif event.delta < 0 and self.font_size > 1:  # 滚轮向下滚动
            self.font_size -= 1
            self.plot_scale_factor /= 1.1  # 缩小线图

        self.update_font_size()
        self.update_plot_scale()

    def update_font_size(self):
        """更新全局字體大小"""
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=self.font_size)

        # 手動更新不自動應用字體的控件
        for widget in self.window.winfo_children():
            if isinstance(widget, (tk.Label, tk.Entry, ttk.Button)):
                widget.config(font=("TkDefaultFont", self.font_size))
    def update_plot_scale(self):
        """根据缩放因子调整线图"""
        if hasattr(self, 'ax') and hasattr(self, 'canvas'):  # 确保图表已初始化
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            # 根据缩放因子调整坐标范围
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_range = (x_max - x_min) / 2 / self.plot_scale_factor
            y_range = (y_max - y_min) / 2 / self.plot_scale_factor

            self.ax.set_xlim([x_center - x_range, x_center + x_range])
            self.ax.set_ylim([y_center - y_range, y_center + y_range])

            self.canvas.draw()
    # _size_slider----------------------
    def disconnect_camera(self):
        if self.connected:
            self.camera_controller.release_camera()
            self.cap = None
            self.camera_controller = None
            self.connected = False
            self.connect_button.config(state='normal')
            self.disconnect_button.config(state='disabled')
            self.device_controller.finalize_device()
            logging.info("Camera disconnected")

        else:
            logging.warning("Camera is not connected")

    def load_config(self):
        config_file = "Dll/config.txt"
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
                    self.width = config.get("width", 2)
                    self.prominence = config.get("prominence", 7.0)
                    self.height = config.get("height", 250)
                    self.lam = config.get("lam", 1000)
                    self.p = config.get("p", 0.001)
                    self.niter = config.get("niter", 5)
                    self.window_length = config.get("window_length", 5)
                    self.polyorder = config.get("polyorder", 3)
                    logging.info("Config loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
                messagebox.showerror("Error", f"Failed to load config: {e}")
        else:
            logging.info("Config file not found, using default settings")
    def initialize_device_and_camera(self):
        self.device_controller = DeviceController()
        if self.device_controller.initialize_device():
            self.device_controller.read_data_from_device()
            self.brightness = self.device_controller.EXP
            self.gain = self.device_controller.gain
        else:
            raise ValueError("No suitable device found")
        self.camera_controller = None
        self.cap = None
        self.connected = False

    def create_widgets(self):
        self.create_buttons()
        self.create_camera_frame()
        self.create_plot_frame()
        self.create_x_axis_options()
        self.initialize_matplotlib()

    def create_buttons(self):
        self.buttons_frame = tk.Frame(self.window)
        self.buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.connect_button = ttk.Button(self.buttons_frame, text="Connect", command=self.connect_camera)
        self.connect_button.pack(side=tk.LEFT, padx=2)

        self.disconnect_button = ttk.Button(self.buttons_frame, text="Disconnect", command=self.disconnect_camera)
        self.disconnect_button.pack(side=tk.LEFT, padx=2)
        self.disconnect_button.config(state='disabled')

        self.setting_button = ttk.Button(self.buttons_frame, text="Setting", command=self.open_setting)
        self.setting_button.pack(side=tk.LEFT, padx=2)

        self.save_button = ttk.Button(self.buttons_frame, text="Save", command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=2)

        self.cubic_fitting_button = ttk.Button(self.buttons_frame, text="Cubic Fitting",
                                               command=self.perform_cubic_fitting)
        self.cubic_fitting_button.pack(side=tk.LEFT, padx=2)

        self.exit_button = ttk.Button(self.buttons_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(side=tk.LEFT, padx=2)

        self.baseline_toggle = ttk.Checkbutton(self.buttons_frame, text="Baseline",
                                               variable=self.baseline_correction)
        self.baseline_toggle.pack(side=tk.LEFT, padx=2)


    def create_camera_frame(self):
        self.camera_frame = ttk.Label(self.window)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=12)

    def create_plot_frame(self):
        self.plot_frame = (tk.Frame
                           (self.window))
        self.plot_frame.grid(row=0, column=1, padx=14, pady=16)

    def create_x_axis_options(self):
        tk.Label(self.buttons_frame, text="X-axis:").pack(side=tk.LEFT, padx=1)
        self.x_axis_option = ttk.Combobox(self.buttons_frame, values=["Pixel", "Wavelength"], state="readonly")
        self.x_axis_option.current(0)
        self.x_axis_option.pack(side=tk.LEFT, padx=2)
        self.x_axis_option.bind("<<ComboboxSelected>>", self.update_plot_x_axis)

    def initialize_matplotlib(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill=tk.BOTH, expand=True)
        self.line, = self.ax.plot([], [], lw=1)

    def update(self):
        if self.connected and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                combined = self.process_frame(frame)
                result = combined.reshape(800, 1280).astype(np.float64)
                normalized_result = (result / 4096 * 254).astype(np.uint8)

                if self.autoROI_var.get():
                    self.update_auto_roi(result)

                start_point = (0, self.device_controller.ROI)
                end_point = (normalized_result.shape[1], self.device_controller.ROI + self.rows_number)
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(normalized_result, start_point, end_point, color, thickness)
                resized_image = cv2.resize(normalized_result, (0, 0), fx=0.4, fy=0.5)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
                self.camera_image_tmp = Image.fromarray(resized_image).copy()
                #logging.info(f"Camera image temp: {self.camera_image_tmp}")
                self.camera_frame.config(image=self.photo)
                self.analyze_roi(result, self.device_controller.ROI, self.rows_number)
        else:
            self.camera_frame.config(image='')
            #self.ax.clear()
            self.canvas.draw()

        self.window.after(self.delay, self.update)

    def update_single(self):
        if self.connected and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                combined = self.process_frame(frame)
                result = combined.reshape(800, 1280).astype(np.float64)
                normalized_result = (result / 4096 * 254).astype(np.uint8)

                if self.autoROI_var.get():
                    self.update_auto_roi(result)

                start_point = (0, self.device_controller.ROI )
                end_point = (normalized_result.shape[1], self.device_controller.ROI + self.rows_number)
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(normalized_result, start_point, end_point, color, thickness)
                resized_image = cv2.resize(normalized_result, (0, 0), fx=0.4, fy=0.5)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
                self.camera_frame.config(image=self.photo)
                self.analyze_roi(result, self.device_controller.ROI, self.rows_number)
        else:
            self.camera_frame.config(image='')
            self.ax.clear()
            self.canvas.draw()

    def update_auto_roi(self, result):
        row_averages = np.mean(result, axis=1)
        max_row_index = np.argmax(row_averages)
        if abs(max_row_index - self.device_controller.ROI) >= 3:
            self.device_controller.ROI = max_row_index
        if hasattr(self, 'roi_value_label') and self.roi_value_label.winfo_exists():
            self.roi_value_label.config(text=str(self.device_controller.ROI))
        if hasattr(self, 'roi_var'):
            self.roi_var.set(self.device_controller.ROI)

    def process_frame(self, frame):
        # 將高位和低位資料提取並組合
        high_bits = frame[0][::2].astype(np.uint16)
        low_bits = frame[0][1::2].astype(np.uint16)
        if len(frame[0]) % 2 != 0:
            low_bits = np.append(low_bits, 0)
        combined_data = ((high_bits << 4) | (low_bits - 128)).astype(np.uint16)
        return combined_data

    def analyze_roi(self, result, ROI, rows_number):
        roi = result[ROI : ROI + rows_number, :]
        roi_avg = np.mean(roi, axis=0)
        self.roi_avg = roi_avg  # Store smoothed roi_avg for cubic fitting
        self.draw_plot(roi_avg)

    def draw_plot(self, data):

        # Plot new data
        if self.baseline_correction.get():
            data = data - np.average(data[0:100])

        x_axis = self.device_controller.x_axis_wavelength if self.x_axis_option.get() == "Wavelength" else self.device_controller.x_axis_pixel
        self.line.set_data(x_axis, data)

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

        with open("Dll/config.txt", "r", encoding="utf-8") as f:
            config = json.load(f)
        THRESHOLD_HIGH = config["threshold_high"]
        THRESHOLD_LOW = config["threshold_low"]
        FEATURE_PARAMS = config["features"]

        self.camera_image_saved = self.camera_image_tmp.copy()
        if self.camera_image_saved is None:
            messagebox.showwarning("Warning", "No camera image available for saving.")

        roi_avg = self.roi_avg.copy()
        if self.baseline_correction.get():
            roi_avg = roi_avg - np.average(roi_avg[0:100])

        data_processor = self.data_processor
        peaks, fwhm_values = data_processor.calculate_fwhm_with_plot(
            roi_avg,
            width=self.width,
            prominence=self.prominence,
            height=self.height,
            baseline_correction=False,
            shift=2
        )
        result = data_processor.cubic_fitting(peaks)
        if result is None or any(v is None for v in result):
            messagebox.showwarning("Warning", "擬合失敗，無法進行後續操作。")
            return

        popt, r_squared, fitted_peaks, matched_wavelengths = result

        baseline_eva = self.data_processor.simple_baseline_evaluation(roi_avg)
        self.baseline_eva = baseline_eva

        result_text = (
            f"Fitting Curve Equation: {popt[0]:.5e}x² + {popt[1]:.5e}x + {popt[2]:.5e}\n"
            f"R²: {r_squared:.3f}\n"
        )
        self.device_controller.WL = [popt[2], popt[1], popt[0], 0]
        print("[DEBUG] self.device_controller.WL =", self.device_controller.WL)

        # Create a new window to display results
        result_window = tk.Toplevel()
        result_window.title("Cubic Fitting Results")

        fig_fit, ax_fit = plt.subplots()
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=result_window)
        canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ax_fit.plot(peaks, matched_wavelengths, 'ro', label='Standard Wavelengths', ms=3)
        ax_fit.plot(peaks, fitted_peaks, 'b--', label='Fitted Curve')
        ax_fit.set_xlabel('Peak Position (pixel)')
        ax_fit.set_ylabel('Standard Wavelength (nm)')
        ax_fit.legend()
        canvas_fit.draw()

        # Display the fitting results in the new window
        fitting_label = tk.Label(result_window, text=result_text)
        fitting_label.pack()

        # Create and update the FWHM table
        self._fwhm_table = self.create_table(result_window)

        # Convert FWHM from pixels to nm using the derivative of the fitting function
        def poly_derivative(x):
            return 2 * popt[0] * x + popt[1]
        FWHM_convert = [poly_derivative(pk) * fw for pk, fw in zip(peaks, fwhm_values)]
        self.update_table_with_fwhm_nm(self._fwhm_table, peaks, matched_wavelengths, FWHM_convert)

        reference_waves = [404.656, 435.833, 546.074, 578.013, 696.543, 763.511, 811.531, 841.807]
        wave_to_fwhm = dict(zip(matched_wavelengths, FWHM_convert))

        # Filter FWHM values based on reference waves
        fwhms_of_interest = []
        for ref_w in reference_waves:
            # Check if the matched wavelength is close to a reference wave
            closest_wave = min(matched_wavelengths, key=lambda x: abs(x - ref_w))
            if abs(closest_wave - ref_w) < 1e-2:  # Tolerance to match closely
                fwhms_of_interest.append(wave_to_fwhm[closest_wave])

        if fwhms_of_interest:
            mean_fwhm = np.mean(fwhms_of_interest)
            std_fwhm = np.std(fwhms_of_interest)
        else:
            mean_fwhm, std_fwhm = None, None

        # Analyze features
        fitting_wavelength = np.array([popt[2] + x * popt[1] + x * x * popt[0] for x in range(1280)])
        spectrum_max = max(roi_avg)

        # Feature 1: Max intensity ratio between 459-530 nm
        feature1 = max(roi_avg[(fitting_wavelength >= 459) & (fitting_wavelength <= 530)]) / spectrum_max

        # Feature 2: Max intensity ratio between 600-668 nm
        feature2 = max(roi_avg[(fitting_wavelength >= 600) & (fitting_wavelength <= 668)]) / spectrum_max

        # Feature 3: Min intensity ratio between 777-792 nm
        feature3 = min(roi_avg[(fitting_wavelength >= 777) & (fitting_wavelength <= 792)]) / spectrum_max

        custom_peak_wavelengths = [435.833, 546.074, 763.511, 811.531]
        feature4_difference = self._calculate_peak_difference(
            roi_avg, fitting_wavelength, custom_peak_wavelengths
        )

        feature5 = (self.brightness / 900) * (self.gain / 32 ) / (spectrum_max / 4000)

        feature6 = max(roi_avg[(fitting_wavelength >= 410) & (fitting_wavelength <= 550)]) / spectrum_max

        feature_values = {
            "fwhm_mean": mean_fwhm,
            "feature1_ratio": feature1,
            "feature2_ratio": feature2,
            "feature3_ratio": feature3,
            "feature4_difference": feature4_difference,
            "feature5": feature5,
            "feature6": feature6,
            "fwhm_std": std_fwhm,
            "Baseline Evaluation": baseline_eva
        }

        def compute_feature_score(value, max_val, min_val, weight):
            if abs(max_val - min_val) < 1e-12:
                return 0.0
            normalized = (max_val - value) / (max_val - min_val)
            return normalized * weight

        total_score = 0.0
        for feature_key, feature_val in feature_values.items():
            max_val = FEATURE_PARAMS[feature_key]["max"]
            min_val = FEATURE_PARAMS[feature_key]["min"]
            weight = FEATURE_PARAMS[feature_key]["weight"]

            score = compute_feature_score(feature_val, max_val, min_val, weight)
            total_score += score

        if total_score >= THRESHOLD_HIGH:
            classification = "A Class"
        elif total_score >= THRESHOLD_LOW:
            classification = "B Class"
        else:
            classification = "C Class"

        result_text += (
            f"Total Score: {total_score * 100:.1f}\n"
            f"Classification: {classification}\n"
        )
        fitting_label.config(text=result_text, font=("Helvetica", 12))


    def _calculate_peak_difference(self, roi_avg, wl_array, custom_peak_wavelengths, half_range=3):
        fitted_list = []
        actual_list = []
        for pw in custom_peak_wavelengths:
            mask = (wl_array >= pw - half_range) & (wl_array <= pw + half_range)
            X_data, Y_data = wl_array[mask], roi_avg[mask]

            if len(X_data) > 0 and len(Y_data) > 0:
                popt_quad, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, X_data, Y_data)
                a, b, c = popt_quad
                peak_intensity = -b / (2*a)
                fitted_list.append(peak_intensity)
                actual_list.append(pw)

        if fitted_list and actual_list:
            return np.mean(np.abs(np.array(fitted_list) - np.array(actual_list)))
        return 0

    def output_data(self,chip_id):
        import csv
        import os
        from datetime import datetime

        if chip_id:  # If chip_id is not an empty string
            # Create a DATA folder and a subfolder with chip_id
            folder_path = os.getcwd()  # Use current working directory as base
            data_folder = os.path.join(folder_path, "DATA")
            os.makedirs(data_folder, exist_ok=True)
            output_dir = os.path.join(data_folder, chip_id)
        else:  # Maintain the current behavior
            # Base folder selection
            from tkinter import filedialog
            folder_path = filedialog.askdirectory(title="Select Directory to Save Files")
            if not folder_path:
                return  # User cancelled the dialog

            folder_name = simpledialog.askstring("Chip ID?", "Enter folder name:")
            if not folder_name:
                folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(folder_path, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save table data to CSV
        table_data = []
        for child in self._fwhm_table.get_children():
            table_data.append(self._fwhm_table.item(child)["values"])
        table_filepath = os.path.join(output_dir, "fwhm_table.csv")
        with open(table_filepath, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Peak Position", "Wavelength (nm)", "FWHM (nm)"])
            writer.writerows(table_data)

        # Save spectrum as an image (PNG)
        image_filepath = os.path.join(output_dir, "spectrum.png")
        self.fig.savefig(image_filepath, dpi=300, format="png")

        # Save spectrum data to CSV
        spectrum_filepath = os.path.join(output_dir, "spectrum_data.csv")
        x_data = self.device_controller.x_axis_wavelength if self.x_axis_option.get() == "Wavelength" else self.device_controller.x_axis_pixel
        y_data = self.ax.lines[0].get_ydata()
        spectrum_data = np.column_stack((x_data, y_data))
        header = "Wavelength (nm),Intensity" if self.x_axis_option.get() == "Wavelength" else "Pixel,Intensity"
        np.savetxt(spectrum_filepath, spectrum_data, fmt="%.6f", header=header, delimiter=",", comments="")

        # Save camera's raw image
        raw_image_filepath = os.path.join(output_dir, "camera_raw_image.png")
        if self.camera_image_saved is None:
            messagebox.showwarning("Warning", "No camera image available for saving.")
        if self.camera_image_saved is not None:
            self.camera_image_saved.save(raw_image_filepath, "PNG")

        # Save UI parameters
        UI_parameters_filepath = os.path.join(output_dir, "UI_parameters.txt")
        parameters = {
            "ROI": int(self.device_controller.ROI),
            "Wavelength (WL)": self.device_controller.WL,
            "Exposure Time (ms)": int(self.brightness),
            "Gain": int(self.gain),
            "Baseline Evaluation": float(self.baseline_eva)
        }
        with open(UI_parameters_filepath, 'w') as file:
            json.dump(parameters, file, indent=4)

        messagebox.showinfo("Success", f"Files saved successfully in {output_dir}")


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

        output_button = ttk.Button(button_frame, text='Output', command=lambda: self.output_data(chip_id=''))
        output_button.pack(side='left')

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
        roi = int(self.device_controller.ROI)
        roi_width = int(self.rows_number)
        wl = self.device_controller.WL
        exp = int(self.brightness)  # Exposure time
        gain = int(self.gain)  # Gain

        # Create a dictionary to hold all information

        data_dict = {
            "version" : 1.0,
            "xts" : 0,
            "motor_rotate_direction" : 0,
            "center_location" :  0,
            "conversion_factor_0_a0" : wl[0],
            "conversion_factor_0_a1" : wl[1],
            "conversion_factor_0_a2" : wl[2],
            "conversion_factor_0_a3" : wl[3],
            "conversion_factor_1_a0": 0,
            "conversion_factor_1_a1": 0,
            "conversion_factor_1_a2": 0,
            "conversion_factor_1_a3": 0,
            "conversion_factor_2_a0": 0,
            "conversion_factor_2_a1": 0,
            "conversion_factor_2_a2": 0,
            "conversion_factor_2_a3": 0,
            "sg_filter_order" : 0,
            "sg_filter_window_size" : 0,
            "motor_pulse_length" : 0,
            "spectrometer_location" : 0,
            "auto_scaling_location" : 0,
            "language" : 0,
            "digital_gain" : 0,
            "analog_gain" : gain,
            "roi" : [0, 1280, roi, 20],
            "init_exp_value" : exp,
            "auto_exp" : 0,
            "auto_exp_lower_limit" : 0,
            "auto_exp_upper_limit" : 0,
            "roi_height" : roi,
            "roi_width" : roi_width,
            "uvcspectro_0_chip_id" : chip_id,
            "uvcspectro_0_serial_nbr" : serial_number,
            "uvcspectro_0_part_nbr": "SPU-M100",
            "uvcspectro_1_chip_id" : 0,
            "uvcspectro_1_serial_nbr" : 0,
            "uvcspectro_1_part_nbr": "SPU-M100",
            "uvcspectro_2_chip_id" : 0,
            "uvcspectro_2_serial_nbr" : 0,
            "spectrodevice_serial_nbr" : 0,
            "spectrodevice_part_nbr": 0,
            "spectrodevice_data_uri_location" : 'http://abontouch.com.tw:8080/UsbSpectrum/php/'
        }

        # Convert the dictionary to a JSON string
        input_data = json.dumps(data_dict, indent=4)

        # Call the method to write data to the device
        success = self.device_controller.write_input_data_to_flash(input_data)
        if success:
            # Save the JSON data to a local file
            local_file_name = f"input_data_{chip_id}_{serial_number}.json"
            try:
                if not os.path.exists('JSON'):
                    os.makedirs('JSON')
                with open(os.path.join('JSON', local_file_name), 'w') as json_file:
                    json_file.write(input_data)
                messagebox.showinfo("Success", f"Write input data: Ok\nSaved locally as {local_file_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save JSON locally: {e}")
        else:
            messagebox.showerror("Error", "Write input data: Fail")
        self.output_data(chip_id= chip_id)
    def update_table_with_fwhm_nm(self, table, peaks, wavelengths, FWHM_convert):
        for i in table.get_children():
            table.delete(i)

        for peak, wavelength, fwhm_nm in zip(peaks, wavelengths, FWHM_convert):
            # print(f"fwhm_nm-> {fwhm_nm}")
            fwhm_nm = abs(fwhm_nm)
            if fwhm_nm is not None and not math.isnan(fwhm_nm) and fwhm_nm <= 40.5:
                table.insert('', 'end', values=(peak, f"{wavelength:.2f}", f"{fwhm_nm:.2f}"))

    def open_setting(self):
        self.setting_window = tk.Toplevel(self.window)
        self.setting_window.title("Settings")

        # ROI setting
        tk.Label(self.setting_window, text="ROI:").grid(row=0, column=0)
        self.roi_scale = tk.Scale(self.setting_window, from_=0, to_=800, orient=tk.HORIZONTAL, variable=self.roi_var)
        self.roi_scale.grid(row=0, column=1)
        self.roi_value_label = tk.Label(self.setting_window, text=str(self.device_controller.ROI))
        self.roi_value_label.grid(row=0, column=2)

        # Auto ROI setting
        self.autoROI_checkbox = tk.Checkbutton(self.setting_window, text="Auto ROI", variable=self.autoROI_var)
        self.autoROI_checkbox.grid(row=0, column=3)

        # Auto Scaling button
        self.auto_scaling_button = ttk.Button(self.setting_window, text="Auto Scaling", command=self.auto_scaling)
        self.auto_scaling_button.grid(row=4, column=3)

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
        self.brightness_scale = tk.Scale(self.setting_window, from_=1, to_=900, orient=tk.HORIZONTAL,
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

        # Close button
        button_frame = tk.Frame(self.setting_window)
        button_frame.grid(row=6, column=0, columnspan=4, pady=10)
        close_button = ttk.Button(button_frame, text="Close", command=self.close_settings)
        close_button.pack(side=tk.LEFT, padx=5)

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
        if self.roi_value_label.winfo_exists():
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
                self.roi_scale.config(state='normal')
                self.device_controller.ROI = self.roi_var.get()
            else:
                # Enable the ROI Scale
                self.roi_scale.config(state='normal')
                self.device_controller.ROI = self.roi_var.get()

        # Redraw the plot
        if hasattr(self, 'roi_avg'):
            self.draw_plot(self.roi_avg)

    def auto_scaling(self):
        self.current_exposure = None
        self.current_gain = None
        self.highest_pixel = None
        self.highest_intensity = None
        self.spectrum = None
        self.auto_scaling_step = 1
        self.sensor_max_value = 4096
        self.current_gain = 1

        self.update_single()
        self.spectrum = self.roi_avg
        if 3250 <= max(self.spectrum) <= self.sensor_max_value - 100:
            logging.info("Auto Scaling process completed. (No need to scale)")
            self.save_config()
            self.auto_scaling_active = False
            return

        def proceed_step():
            if self.auto_scaling_step == 1:  # Step 1: 设置 Exposure 为 100 和 Gain 为 2
                self.current_exposure = 100
                self.current_gain += 1
                self.brightness_var.set(self.current_exposure)
                self.gain_var.set(self.current_gain)
                logging.info(
                    f"Step {self.auto_scaling_step}: Set Exposure to {self.current_exposure} and Gain to {self.current_gain}")
                self.update_settings()
                self.auto_scaling_step += 1
                self.update_single()
                self.window.after(500, proceed_step)


            elif self.auto_scaling_step == 2:
                self.spectrum = self.roi_avg
                if self.spectrum is None:
                    logging.warning("Spectrum data not ready. Waiting...")
                    self.update_single()
                    self.window.after(500, proceed_step)
                    return
                self.highest_pixel = int(np.argmax(self.spectrum))
                self.highest_intensity = float(self.spectrum[self.highest_pixel])
                logging.info(
                    f"Step {self.auto_scaling_step}: Highest pixel = {self.highest_pixel}, Intensity = {self.highest_intensity}")
                self.auto_scaling_step += 1
                self.update_single()
                self.window.after(200, proceed_step)

            elif self.auto_scaling_step == 3:  # Step 3: 设置 Exposure 为 300 和 Gain 为 1
                self.current_exposure = 300
                # self.current_gain = 1
                self.brightness_var.set(self.current_exposure)
                self.gain_var.set(self.current_gain)
                logging.info(
                    f"Step {self.auto_scaling_step}: Set Exposure to {self.current_exposure} and Gain to {self.current_gain}")
                self.update_settings()
                self.auto_scaling_step += 1
                self.update_single()
                self.window.after(500, proceed_step)

            elif self.auto_scaling_step == 4:  # Step 4: 记录第 2 步中最高点的 pixel 对应的强度
                self.spectrum = self.roi_avg  # 获取最新光谱数据
                if self.spectrum is None:
                    logging.warning("Spectrum data not ready. Waiting...")
                    self.update_single()
                    self.window.after(500, proceed_step)  # 等待数据更新
                    print("hello")
                    return
                pixel_intensity = float(self.spectrum[self.highest_pixel])
                logging.info(
                    f"Step {self.auto_scaling_step}: Pixel {self.highest_pixel} intensity at Exposure {self.current_exposure} = {pixel_intensity}")
                self.pixel_intensity_at_300 = pixel_intensity  # 保存强度
                self.auto_scaling_step += 1
                self.update_single()
                self.window.after(1000, proceed_step)

            elif self.auto_scaling_step == 5:  # Step 5: 计算目标 Exposure
                target_intensity = self.sensor_max_value - 800  # 目标强度
                target_exposure = int((self.current_exposure * target_intensity) / self.pixel_intensity_at_300)

                if target_exposure >= 800:  # 最大exposure
                    self.auto_scaling_step = 1
                    return

                self.current_exposure = target_exposure
                self.brightness_var.set(self.current_exposure)
                logging.info(
                    f"Step {self.auto_scaling_step}: Target Exposure calculated as {target_exposure}, setting now")
                self.update_settings()
                self.auto_scaling_step += 1
                self.update_single()
                self.window.after(500, proceed_step)

            elif self.auto_scaling_step == 6:  # Step 6: 检查第 2 步的最高点强度是否在范围内

                self.spectrum = self.roi_avg  # 获取最新光谱数据

                if self.spectrum is None:
                    logging.warning("Spectrum data not ready. Waiting...")
                    self.update_single()

                    self.window.after(500, proceed_step)  # 等待数据更新

                    return

                # final_intensity = float(self.spectrum[self.highest_pixel])
                final_intensity = max(self.spectrum)

                if 3250 <= final_intensity <= self.sensor_max_value - 100:

                    logging.info(
                        f"Step {self.auto_scaling_step}: Pixel {self.highest_pixel} intensity {final_intensity} is within range (3250, {self.sensor_max_value - 100})"
                    )

                    logging.info("Auto Scaling process completed.")
                    self.save_config()

                    self.auto_scaling_active = False  # 流程结束

                else:

                    if final_intensity < 3250:

                        self.current_exposure = min(self.current_exposure + 10,
                                                    self.sensor_max_value)  # 增加曝光值，避免超过传感器限制

                        adjustment = "increased"

                    elif final_intensity > self.sensor_max_value - 100:

                        self.current_exposure = max(self.current_exposure - 10, 1)  # 减少曝光值，确保不低于 1

                        adjustment = "decreased"

                    self.brightness_var.set(self.current_exposure)

                    logging.warning(

                        f"Step {self.auto_scaling_step}: Pixel {self.highest_pixel} intensity {final_intensity} is out of range! Exposure {adjustment} to {self.current_exposure}"

                    )

                    self.update_settings()
                    self.update_single()
                    self.window.after(500, proceed_step)  # 延时后重新检查

        # 启动流程
        self.auto_scaling_active = True
        proceed_step()

    def close_settings(self):
        """Execute save_config and close the settings window."""
        self.save_config()
        self.setting_window.destroy()
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
        self.device_controller.finalize_device()
        if self.connected and self.camera_controller:
            self.camera_controller.release_camera()
        self.window.quit()
        self.window.destroy()

    def save_config(self):
        config_file = "Dll/config.txt"

        # 嘗試讀取現有的配置檔案
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as file:
                    config = json.load(file)
            except Exception as e:
                logging.error(f"Failed to read existing config: {e}")
                config = {}
        else:
            config = {}

        config.update({
            "x_axis_min": self.x_axis_min_var.get(),
            "x_axis_max": self.x_axis_max_var.get(),
            "rows_number": self.rows_number,
            "brightness": self.brightness,
            "gain": self.gain,
            "baseline_correction": self.baseline_correction.get(),
            "autoROI": self.autoROI_var.get(),
            "ROI": self.roi_var.get()
        })
        try:
            with open(config_file, "w") as file:
                json.dump(config, file, indent=4)
            logging.info("Config saved successfully")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            messagebox.showerror("Error", f"Failed to save config: {e}")

# Create window and start application
if __name__ == "__main__":
    app = App(tk.Tk(), "ROI Monitor")
