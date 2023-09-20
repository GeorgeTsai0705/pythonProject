import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

Spectrum = np.array([1, 2, 3, 4, 5])  # 設定預設值
Standard_Wavelength = np.array([365.34, 406.15, 436.00, 545.79, 578.60, 696.56, 706.58, 727.17,
                                738.34, 750.66, 763.56, 772.34, 794.56, 800.98, 811.48, 826.63,
                                842.33, 912.38, 922.18])

def read_numeric_data(filename):
    data = []

    with open(filename, 'r') as file:
        for line in file:
            # 移除行尾的換行符並將其拆分為單個數字
            numbers = line.strip().split()

            # 將數字轉換為浮點數並添加到資料列表中
            data.extend([float(num) for num in numbers])

    return np.array(data)

def open_image():
    img_path = "HgArSpectralExample.png"
    img = Image.open(img_path)
    img.show()

def calculate_fwhm(spectrum, width, prominence):
    # 尋找峰值位置
    peaks, _ = find_peaks(spectrum[0:1000], width=width, prominence=prominence)

    # 計算每個峰值的半高全寬
    fwhm_values = []
    for peak in peaks:
        # 獲取峰值的高度
        peak_height = spectrum[peak]

        # 尋找半高的位置
        half_height = peak_height / 2.0

        # 在波峰範圍內搜索半高
        left_idx = peak - np.argmax(spectrum[peak::-1] <= half_height)
        right_idx = peak + np.argmax(spectrum[peak:] <= half_height)

        # 計算半高全寬
        fwhm = round((right_idx - left_idx) * 0.75, 1)
        fwhm_values.append(fwhm)

    # 返回峰值位置及其相應的半高全寬值
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

    # 尋找峰值位置
    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)
    peaks, _ = calculate_fwhm(Spectrum, width, prominence)

    # 获取选中的开关变量的值
    switch_vars = [var.get() for var in switch_vars1 + switch_vars2]
    print(switch_vars)

    if len(peaks) < sum(switch_vars):
        fitting_label.config(text="峰值數量不足，無法進行拟合")
        correlation_label.config(text="")
        return

    # 根据选中的开关变量的值，提取对应的Standard_peak值

    selected_standard_peak = [peak for peak, switch in zip(Standard_Wavelength, switch_vars) if switch != 0]
    Standard_peak = selected_standard_peak
    # 只保留选中的波长和强度值
    Spectrum_peak = peaks[:len(selected_standard_peak)]

    # 執行一元二次拟合
    coefficients = np.polyfit(Spectrum_peak, Standard_peak, 2)
    f = np.poly1d(coefficients)

    # 計算相關係數
    correlation = np.corrcoef(Standard_peak, f(Spectrum_peak))[0, 1]

    # 顯示拟合曲線方程式與相關係數
    fitting_label.config(text="拟合曲線方程式: {}".format(f))
    correlation_label.config(text="相關係數: {:.4f}".format(correlation))

    # 更新拟合曲線圖形
    ax_fit.cla()
    ax_fit.plot(Spectrum_peak, Standard_peak, 'bo', label='Points',ms=3)
    ax_fit.plot(Spectrum_peak, f(Spectrum_peak), 'b--', label='Fitted Curve')
    ax_fit.set_xlabel('Standard_peak (nm)')
    ax_fit.set_ylabel('Spectrum_peak (pixel)')
    ax_fit.legend()
    canvas_fit.draw()

def update_plot_and_results(Spectrum):
    # 讀取滑動條的參數值
    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)

    peaks, fwhm_values = calculate_fwhm(Spectrum, width, prominence)

    # 清除圖形
    ax.cla()

    # 繪製光譜圖
    ax.plot(Spectrum)

    # 標記峰值位置
    ax.plot(peaks, Spectrum[peaks], 'ro')  # 在峰值位置處繪製紅色圓圈

    # 更新坐標軸標籤
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')

    # 顯示圖形
    canvas.draw()

    # 更新峰值結果
    result_label.config(text="峰值位置: {}\n半高全寬值: {}".format(peaks, fwhm_values))

def update_width(event):
    update_plot_and_results(Spectrum)

def update_prominence(event):
    update_plot_and_results(Spectrum)

def terminate_program():
    sys.exit()

# 創建主窗口
root = tk.Tk()
root.title("可讀性程式")

# 設置Grid布局
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# 創建左邊的圖形框架
left_frame = tk.Frame(root)
left_frame.grid(row=0, column=0, sticky="nsew")
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

# 創建右邊的圖形框架
right_frame = tk.Frame(root)
right_frame.grid(row=0, column=1, sticky="nsew")
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

# 創建圖形窗口
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 創建峰值結果標籤
result_label = tk.Label(left_frame, text="")
result_label.pack()

# 創建滑動條和標籤框架
scale_frame = tk.Frame(left_frame)
scale_frame.pack()

# 創建滑動條和標籤
width_label = tk.Label(scale_frame, text="Width: ")
width_label.pack(side=tk.LEFT)
width_scale = tk.Scale(scale_frame, from_=0, to=10, resolution=0.1, length=200, orient=tk.HORIZONTAL,
                       command=update_width)
width_scale.set(1.5)
width_scale.pack(side=tk.LEFT, padx=5)

prominence_label = tk.Label(scale_frame, text="Prominence: ")
prominence_label.pack(side=tk.LEFT)
prominence_scale = tk.Scale(scale_frame, from_=0, to=20, resolution=0.1, length=200, orient=tk.HORIZONTAL,
                            command=update_prominence)
prominence_scale.set(6)
prominence_scale.pack(side=tk.LEFT, padx=5)

# 修改開關框架
switch_frame = tk.Frame(left_frame)
switch_frame.pack()

# 名稱列表
names = ['365', '405', '436', '545', '578', '696', '706', '727', '738', '750', '763', '772', '794', '800', '811', '826',
         '842', '912', '922']

# 拆分名稱列表成兩個部分
names1 = names[:10]
names2 = names[10:]

# 創建開關框架1
switch_frame1 = tk.Frame(switch_frame)
switch_frame1.pack(side=tk.TOP, padx=10, pady=10)

# 創建開關框架2
switch_frame2 = tk.Frame(switch_frame)
switch_frame2.pack(side=tk.BOTTOM, padx=10, pady=10)

# 創建開關
switch_vars1 = []  # 存儲開關的變量
switches1 = []  # 存儲開關
for name in names1:
    var = tk.IntVar()  # 創建IntVar變量
    var.set(1)  # 設置默認值為1，表示"ON"
    switch_vars1.append(var)  # 將變量添加到列表中
    switch = tk.Checkbutton(switch_frame1, text=name, variable=var)
    switch.pack(side=tk.LEFT)
    switches1.append(switch)

# 創建開關
switch_vars2 = []  # 存儲開關的變量
switches2 = []  # 存儲開關
for name in names2:
    var = tk.IntVar()  # 創建IntVar變量
    var.set(1)  # 設置默認值為1，表示"ON"
    switch_vars2.append(var)  # 將變量添加到列表中
    switch = tk.Checkbutton(switch_frame2, text=name, variable=var)
    switch.pack(side=tk.LEFT)
    switches2.append(switch)

# 創建按鈕框架
button_frame = tk.Frame(left_frame)
button_frame.pack()

# 創建"Open"按鈕
open_button = tk.Button(button_frame, text='Open', command=open_file)
open_button.pack(side=tk.LEFT, padx=5)

# 創建"Fitting"按鈕
fitting_button = tk.Button(button_frame, text='Fitting', command=perform_fitting)
fitting_button.pack(side=tk.LEFT, padx=5)

# 在button_frame裡面添加新按鈕
image_button = tk.Button(button_frame, text='Open Image', command=open_image)
image_button.pack(side=tk.LEFT, padx=5)

# 創建"Terminate"按鈕
terminate_button = tk.Button(button_frame, text='Terminate', command=terminate_program)
terminate_button.pack(side=tk.LEFT, padx=5)

# 創建右邊的圖形窗口
fig_fit, ax_fit = plt.subplots()
canvas_fit = FigureCanvasTkAgg(fig_fit, master=right_frame)
canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 創建峰值和半高全寬的Label
fitting_label = tk.Label(right_frame, text="")
fitting_label.pack()

# 創建相關係數標籤
correlation_label = tk.Label(right_frame, text="")
correlation_label.pack()

def update_correlation_label(correlation):
    correlation_label.config(text="相關係數: {:.4f}".format(correlation))

# 執行主迴圈
root.mainloop()
