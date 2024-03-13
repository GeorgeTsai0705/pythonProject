import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

Spectrum = np.array([1, 2, 3, 4, 5])  # 設定預設值
Standard_Wavelength = np.array([365.34, 406.15, 436.00, 545.79, 578.60, 696.56, 706.58, 727.17,
                                738.34, 750.66, 763.56, 772.34, 794.56, 800.98, 811.48, 826.63,
                                842.33, 852.20, 912.38, 922.18])


def read_numeric_data(filename):
    data = []
    has_header = False
    intensity_index = 0  # Default to the first column

    with open(filename, 'r') as file:
        # Read the first line to check for a header
        first_line = file.readline().strip().split()

        # Check if the first row contains string values (indicating a header)
        if any(element.isalpha() for element in first_line):
            has_header = True
            # Find the index of the "Intensity" column
            if "Intensity" in first_line:
                intensity_index = first_line.index("Intensity")
            else:
                raise ValueError("No column labeled 'Intensity' found in header.")

        # If there's a header, continue reading the file from the second line
        if has_header:
            for line in file:
                elements = line.strip().split()
                data.append(float(elements[intensity_index]))
        else:
            # If there's no header, use the first line and continue reading the file
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
    # 將 height 從字符串轉換為浮點數
    try:
        height_value = float(height)
    except ValueError:
        height_value = None  # 如果輸入無效，則使用 None

    # 尋找峰值位置
    peaks, _ = find_peaks(spectrum[0:1000], width=width, prominence=prominence, height=height_value, distance=6)

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
        fwhm = round((right_idx - left_idx), 1)
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

    # Find peaks
    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)
    height = round(float(height_entry.get()), 1)
    peaks, fwhm_values = calculate_fwhm(Spectrum, width, prominence, height)

    # Get switch variables
    switch_vars = [var.get() for var in switch_vars1 + switch_vars2]

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

    # Calculate correlation coefficient
    correlation = np.corrcoef(Standard_peak, f(Spectrum_peak))[0, 1]

    # Update fitting curve equation and correlation coefficient
    fitting_label.config(text=f"Fitting Curve Equation: {f}")
    correlation_label.config(text=f"Correlation Coefficient: {correlation:.4f}")

    # 更新拟合曲線圖形
    ax_fit.cla()
    ax_fit.plot(Spectrum_peak, Standard_peak, 'ro', label='Points',ms=3)
    ax_fit.plot(Spectrum_peak, f(Spectrum_peak), 'b--', label='Fitted Curve')
    ax_fit.set_xlabel('Standard_peak (nm)')
    ax_fit.set_ylabel('Spectrum_peak (pixel)')
    ax_fit.legend()
    canvas_fit.draw()

    # Calculate FWHM_convert for each peak and multiply by corresponding FWHM
    FWHM_convert = [(coefficients[1] + coefficients[0] * peak *2) * fwhm for peak, fwhm in zip(peaks, fwhm_values)]

    # Update the table with FWHM(nm) values
    update_table_with_fwhm_nm(results_table, peaks, fwhm_values, FWHM_convert)

    return FWHM_convert

def update_table_with_fwhm_nm(table, peaks, fwhm_values, FWHM_convert):
    # Clear the old data from the table
    for i in table.get_children():
        table.delete(i)

    # Add new data with FWHM(nm) values
    for peak, fwhm, fwhm_nm in zip(peaks, fwhm_values, FWHM_convert):
        table.insert('', 'end', values=(peak, fwhm, round(fwhm_nm, 1)))

# 在 GUI 中創建 Treeview 控件的函數
def create_table(parent):
    # 創建表格及其滾動條
    tree = ttk.Treeview(parent, columns=('Peak Position', 'FWHM', 'FWHM(nm)'), show='headings')
    scrollbar = ttk.Scrollbar(parent, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # 設置列標題
    tree.heading('Peak Position', text='峰值位置')
    tree.heading('FWHM', text='FWHM\n(Pixel)')
    tree.heading('FWHM(nm)', text='FWHM\n(nm)')

    # 設置列對齊方式
    tree.column('Peak Position', anchor='center', width=60)
    tree.column('FWHM', anchor='center', width=60)
    tree.column('FWHM(nm)', anchor='center', width=60)

    # 布局
    tree.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    return tree

# 更新表格數據的函數
def update_table(table, peaks, fwhm_values):
    # 清除表格中的舊數據
    for i in table.get_children():
        table.delete(i)

    # 添加新數據
    for peak, fwhm in zip(peaks, fwhm_values):
        table.insert('', 'end', values=(peak, fwhm))


def update_plot_and_results(Spectrum):
    # 讀取滑動條的參數值
    width = round(float(width_scale.get()), 1)
    prominence = round(float(prominence_scale.get()), 1)

    peaks, fwhm_values = calculate_fwhm(Spectrum, width, prominence, height_entry.get())

    # 清除圖形
    ax.cla()

    # 繪製光譜圖
    ax.plot(Spectrum)

    # 標記峰值位置
    ax.plot(peaks, Spectrum[peaks], 'ro')  # 在峰值位置處繪製紅色圓圈

    # 更新坐標軸標籤
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity')

    # 顯示圖形
    canvas.draw()

    # 更新表格數據
    update_table(results_table, peaks, fwhm_values)

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
results_table = create_table(left_frame)  # 假設您想將其放在 left_frame 中

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
width_scale = tk.Scale(scale_frame, from_=0, to=10, resolution=0.1, length=150, orient=tk.HORIZONTAL,
                       command=update_width)
width_scale.set(1.5)
width_scale.pack(side=tk.LEFT, padx=5)

prominence_label = tk.Label(scale_frame, text="Prominence: ")
prominence_label.pack(side=tk.LEFT)
prominence_scale = tk.Scale(scale_frame, from_=0, to=20, resolution=0.1, length=150, orient=tk.HORIZONTAL,
                            command=update_prominence)
prominence_scale.set(6)
prominence_scale.pack(side=tk.LEFT, padx=5)

# 在 scale_frame 中添加 height 的輸入框
height_label = tk.Label(scale_frame, text="Height: ")
height_label.pack(side=tk.LEFT, pady=5)  # 調整對齊，使其與滑動條一致

# 創建一個 StringVar 實例
height_var = tk.StringVar()
height_var.set("10")  # 設置預設值為 10

# 使用 StringVar 創建 Entry 控件，並設置適當的寬度
height_entry = tk.Entry(scale_frame, textvariable=height_var, width=5)
height_entry.pack(side=tk.LEFT, padx=5, pady=5)  # 添加 pady 參數以進行垂直對齊

# 修改開關框架
switch_frame = tk.Frame(left_frame)
switch_frame.pack()

# 名稱列表
names = ['365', '405', '436', '545', '578', '696', '706', '727', '738', '750', '763', '772', '794', '800', '811', '826',
         '842', '852', '912', '922']

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
