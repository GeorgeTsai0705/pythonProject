import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.ROI = 500
        self.rows_number = 10
        self.brightness = 30
        self.gain = 2

        # Initialize the camera
        self.cap = self.find_and_initialize_camera()
        if not self.cap:
            raise ValueError("No suitable camera found")

        # 設置攝像頭參數
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        self.cap.set(cv2.CAP_PROP_MODE, 2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        # Loading parameters
        parameters = self.read_parameters()
        self.a0 = parameters.get('a0')
        self.a1 = parameters.get('a1')
        self.a2 = parameters.get('a2')
        self.a3 = parameters.get('a3')
        self.x_axis = [self.a0 + x * self.a1 + x * x * self.a2 + x * x * x * self.a3 for x in range(1280)]

        # 創建界面布局
        self.create_widgets()
        self.delay = parameters.get('Delay')
        self.update()
        self.window.mainloop()

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
        self.buttons_frame = tk.Frame(self.window)
        self.buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.setting_button = ttk.Button(self.buttons_frame, text="Setting", command=self.open_setting)
        self.setting_button.pack(side=tk.LEFT, padx=2)

        self.save_button = ttk.Button(self.buttons_frame, text="Save", command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=2)

        self.exit_button = ttk.Button(self.buttons_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(side=tk.LEFT, padx=2)

        # 創建左側畫面（攝像頭原始畫面）
        self.camera_frame = ttk.Label(self.window)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=14)

        # 創建右側畫面（ROI分析結果）
        self.plot_frame = tk.Frame(self.window)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=14)

        # Initialize Matplotlib figure and axes for plotting
        self.fig, self.ax = plt.subplots(figsize=(4, 3.8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill=tk.BOTH, expand=True)
        self.ax.set_xlim([300, 1000])
        self.ax.set_ylim([0, 4100])
        self.ax.set_title("Real-time Spectrum")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Average Value")
        self.line, = self.ax.plot([], [])

    def read_parameters(self):
        with open('Parameter.txt', 'r') as file:
            parameters = json.load(file)
        return parameters

    def update(self):
        # start_time = time.time()
        # print(time.time())
        ret, frame = self.cap.read()
        # 如果讀取成功，更新左側畫面
        if ret:

            a = frame[0][::2].astype(np.uint16)  # 從0開始，每2個元素取一個
            b = frame[0][1::2].astype(np.uint16)  # 從1開始，每2個元素取一個
            if len(frame[0]) % 2 != 0:
                b = np.append(b, 0)  # 確保a和b的長度相同

            combined = ((a << 4) | (b - 128)).astype(np.uint16)

            result = combined.reshape(800, 1280)

            normalized_result = np.clip(result * np.float32(0.0619), 0, 255)

            start_point = (0, self.ROI - self.rows_number)  # 起始点坐标
            end_point = (normalized_result.shape[1], self.ROI + self.rows_number)  # 终点坐标
            color = (255, 0, 0)  # BGR颜色值，红色
            thickness = 2  # 线条厚度

            # 在resized_image上绘制红框
            cv2.rectangle(normalized_result, start_point, end_point, color, thickness)
            resized_image = cv2.resize(normalized_result, (0, 0), fx=0.4, fy=0.4)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            self.camera_frame.config(image=self.photo)
            self.analyze_roi(result, self.ROI, self.rows_number)
            self.window.after(self.delay, self.update)
        # loop_time = time.time() - start_time
        # print(f"Loop time: {loop_time:.4f} seconds")

    def analyze_roi(self, result, ROI, rows_number):
        roi_start = max(0, ROI - rows_number)
        roi_end = min(result.shape[0], ROI + rows_number)
        roi = result[roi_start:roi_end, :]
        roi_avg = np.mean(roi, axis=0)
        self.draw_plot(roi_avg)

    def draw_plot(self, data):
        self.line.set_data(self.x_axis, data)
        self.canvas.draw_idle()

    def open_setting(self):
        self.setting_window = tk.Toplevel(self.window)
        self.setting_window.title("Settings")

        # ROI设置
        tk.Label(self.setting_window, text="ROI:").grid(row=0, column=0)
        self.roi_var = tk.IntVar(value=self.ROI)
        self.roi_scale = tk.Scale(self.setting_window, from_=0, to_=900, orient=tk.HORIZONTAL, variable=self.roi_var,
                                  command=self.update_roi_display)
        self.roi_scale.grid(row=0, column=1)
        self.roi_value_label = tk.Label(self.setting_window, text=str(self.ROI))
        self.roi_value_label.grid(row=0, column=2)

        # Rows Number设置
        tk.Label(self.setting_window, text="Rows Number:").grid(row=1, column=0)
        self.rows_number_var = tk.IntVar(value=self.rows_number)
        self.rows_number_scale = tk.Scale(self.setting_window, from_=1, to_=30, orient=tk.HORIZONTAL,
                                          variable=self.rows_number_var, command=self.update_rows_number_display)
        self.rows_number_scale.grid(row=1, column=1)
        self.rows_number_value_label = tk.Label(self.setting_window, text=str(self.rows_number))
        self.rows_number_value_label.grid(row=1, column=2)

        # Brightness setting
        tk.Label(self.setting_window, text="Exposure:").grid(row=2, column=0)
        self.brightness_var = tk.IntVar(value=self.brightness)
        self.brightness_scale = tk.Scale(self.setting_window, from_=0, to_=800, orient=tk.HORIZONTAL,
                                         variable=self.brightness_var, command=self.update_brightness_display)
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

    def save_data(self):
        # 提示用戶選擇儲存路徑並保存數據
        filepath = filedialog.asksaveasfilename(defaultextension=".txt")
        if filepath:
            ydata = self.ax.lines[0].get_ydata()
            data_to_save = np.column_stack((self.x_axis, ydata))
            np.savetxt(filepath, data_to_save, fmt='%f')

    def exit_app(self):
        self.window.quit()
        self.window.destroy()


# 創建窗口並啟動應用
app = App(tk.Tk(), "ROI Monitor")
