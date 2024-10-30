import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.ROI = 500  # 初始ROI值
        self.rows_number = 10  # 初始行數值
        self.brightness = 17  # 初始Brightness數值
        self.gain = 2 # 初始Gain數值

        self.measurement_data = []  # List to store measurement data
        self.collecting_data = False  # Flag to check if data collection is active

        # 初始化攝像頭
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("無法打開攝像頭")

        # 設置攝像頭參數
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        self.cap.set(cv2.CAP_PROP_MODE, 2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        #self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 17)
        #self.cap.set(cv2.CAP_PROP_GAIN, 8)

        # 創建界面布局
        self.create_widgets()

        # 定時器
        self.delay = 50
        self.update()

        self.window.mainloop()

    def create_widgets(self):
        # 創建按鈕
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
        self.camera_frame.grid(row=0, column=0, padx=10, pady=12)

        # 創建右側畫面（ROI分析結果）
        self.plot_frame = tk.Frame(self.window)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=12)

        # Initialize Matplotlib figure and axes for plotting
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill=tk.BOTH, expand=True)

    def update(self):
        # 從攝像頭讀取畫面
        ret, frame = self.cap.read()
        # 如果讀取成功，更新左側畫面
        if ret:
            a = frame[0][::2].astype(np.uint16)  # 從0開始，每2個元素取一個
            b = frame[0][1::2].astype(np.uint16)  # 從1開始，每2個元素取一個
            if len(frame[0]) % 2 != 0:
                b = np.append(b, 0)  # 確保a和b的長度相同

            combined = ((a << 4) | (b - 128)).astype(np.uint16)
            result = combined.reshape(800, 1280).astype(np.float64)
            normalized_result = (result / 4096 * 254).astype(np.uint8)

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

            if self.collecting_data:
                self.collect_measurement(result)

            self.window.after(self.delay, self.update)

    def analyze_roi(self, result, ROI, rows_number):
        roi = result[ROI - rows_number: ROI + rows_number, :]
        roi_avg = np.mean(roi, axis=0)
        self.draw_plot(roi_avg)

    def draw_plot(self, data):

        # Clear the previous data
        self.ax.clear()

        # Plot new data
        self.ax.plot(data)
        self.ax.set_xlim([0, 1280])  # Set x-axis limits
        self.ax.set_ylim([0, 4100])  # Set y-axis limits
        self.ax.set_title("Real-time Spectrum")
        self.ax.set_xlabel("Wavelength (pixel)")
        self.ax.set_ylabel("Average Value")

        self.canvas.draw()

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
        self.brightness_scale = tk.Scale(self.setting_window, from_=0, to_=800, orient=tk.HORIZONTAL, variable=self.brightness_var,
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
        # Initialize data collection
        self.measurement_data = []  # Clear any previous data
        self.collecting_data = True  # Start collecting data

    def collect_measurement(self, result):
        # Collect measurements
        roi = result[self.ROI - self.rows_number: self.ROI + self.rows_number, :]
        roi_avg = np.mean(roi, axis=0)
        self.measurement_data.append(roi_avg)

        if len(self.measurement_data) >= 100:
            # Stop collecting data
            self.collecting_data = False

            # Transpose the collected data
            transposed_data = np.transpose(self.measurement_data)

            # Save the collected data to a file
            filepath = filedialog.asksaveasfilename(defaultextension=".txt")
            if filepath:
                np.savetxt(filepath, np.array(transposed_data), fmt='%f')

    def exit_app(self):
        self.window.quit()
        self.window.destroy()


# 創建窗口並啟動應用
app = App(tk.Tk(), "ROI Monitor")
