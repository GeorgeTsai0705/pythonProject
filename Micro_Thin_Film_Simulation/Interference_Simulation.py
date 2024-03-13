import tkinter as tk
from tkinter import ttk, filedialog
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Initialization of global variables
wavelengths, original_intensities, simulation_results, silicon_refractive_index = None, None, None, None

# Load initial data from file
file_path = 'Original_Fiber.txt'

def load_data_from_file():
    """Load wavelength and intensity data from a file."""
    global wavelengths, original_intensities, silicon_refractive_index
    try:
        wavelengths, original_intensities, silicon_refractive_index= np.loadtxt(file_path, unpack=True)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")

load_data_from_file()

# Optical simulation functions
def simulate(n0, n1, n2, d, theta0, l):
    """Calculate the interference pattern."""
    cos_theta1 = math.sqrt(1-(n0/n1*math.sin(theta0))**2)
    part1 = r_ij(n0, n1)**2
    part2 = (r_ij(n1, n2)**2) * ((1 - r_ij(n0, n1)**2)**2)
    part3 = 2 * r_ij(n0, n1) * r_ij(n1, n2) * (1 - r_ij(n0, n1)**2) * math.cos(4 * math.pi * n1 * d * cos_theta1 / l)
    return part1 + part2 + part3

def r_ij(ni, nj):
    """Calculate the reflection coefficient."""
    return (ni - nj) / (ni + nj)

def export_data():
    # Assuming 'wavelengths' and 'simulation_results' hold your XY data
    # You might need to adjust this to fit how you manage data in your update_simulation function
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        with open(file_path, 'w') as file:
            for w, r in zip(wavelengths, simulation_results):
                file.write(f"{w}\t{r}\n")
        result_var.set(f"Data exported to {file_path}")

def update_simulation(value=None, delayed=True):
    global wavelengths, simulation_results, silicon_refractive_index   # 聲明為全局變量

    # Cancel any existing scheduled update
    if hasattr(update_simulation, "after_id") and update_simulation.after_id:
        root.after_cancel(update_simulation.after_id)
        update_simulation.after_id = None

    def actual_update():
        global wavelengths, simulation_results, silicon_refractive_index  # 在這裡再次聲明，以確保這些變量是全局的
        n0 = n0_slider.get()
        n1 = n1_slider.get()
        d = d_slider.get()
        try:
            theta1 = math.radians(float(theta1_entry.get()))
        except ValueError:
            result_var.set("Please enter a valid value for theta1")
            return
        simulation_results = [simulate(n0, n1, n2, d, theta1, l) for n2, l in zip(silicon_refractive_index,wavelengths)]

        # Clear the figure for fresh plotting
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(wavelengths, simulation_results, '-o', markersize=1)
        ax.set_title('Simulation Results over Wavelength')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Simulation Result')
        ax.set_xlim([600,800])
        canvas.draw()



    # Schedule the update if delayed, else run immediately
    if delayed:
        update_simulation.after_id = root.after(500, actual_update)  # Adjust the delay as needed
    else:
        actual_update()


def load_and_process_data():
    global original_intensities, silicon_refractive_index
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        # 讀取檔案
        wavelengths, original_intensities = np.loadtxt(file_path, unpack=True)

        # 進行模擬計算
        try:
            theta1 = math.radians(float(theta1_entry.get()))
        except ValueError:
            result_var.set("Please enter a valid value for theta1")
            return

        n0 = n0_slider.get()
        n1 = n1_slider.get()
        n2 = n2_slider.get()
        d = d_slider.get()

        simulated_intensities = [simulate(n0, n1, n2, d, theta1, l) for n2,l in zip(silicon_refractive_index,wavelengths)]

        # 計算 new_intensity
        new_intensities = original_intensities * simulated_intensities

        # 更新第二個圖形
        fig2.clf()  # 清除之前的繪圖
        ax2 = fig2.add_subplot(111)
        ax2.plot(wavelengths, original_intensities, '-', label='Original Intensity', markersize=1)
        ax2.plot(wavelengths, new_intensities, '-', label='New Intensity', markersize=1)
        ax2.legend()
        ax2.set_title('Intensity Comparison')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Intensity')
        ax2.set_xlim([600,800])
        canvas2.draw()

def update_fig2_only():
    global wavelengths,original_intensities, target_intensities, silicon_refractive_index  # 確保可以訪問全局變量

    try:
        theta1 = math.radians(float(theta1_entry.get()))  # 從輸入獲取theta1並轉換為弧度
    except ValueError:
        result_var.set("Please enter a valid value for theta1")  # 錯誤處理
        return

    n0 = n0_slider.get()
    n1 = n1_slider.get()
    d = d_slider.get()

    simulated_intensities = [simulate(n0, n1, n2, d, theta1, l) for n2,l in zip(silicon_refractive_index,wavelengths)]  # 計算模擬強度
    new_intensities = original_intensities * np.array(simulated_intensities)  # 計算新的強度值

    new_intensities_normalized = (new_intensities - np.min(new_intensities)) / np.max(new_intensities)
    target_intensities_normalized = (target_intensities - np.min(target_intensities)) / np.max(target_intensities)

    # 更新第二個圖形
    fig2.clf()  # 清除之前的繪圖
    ax2 = fig2.add_subplot(111)
    ax2.plot(wavelengths, target_intensities_normalized, '-', label='Target Intensity', markersize=1)
    ax2.plot(wavelengths, new_intensities_normalized, '-', label='New Intensity', markersize=1)
    ax2.legend()
    ax2.set_title('Intensity Comparison')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Intensity')
    ax2.set_xlim([600,800])
    canvas2.draw()

# 創建滑動條和標籤的函數，並允許指定一個預設值
def create_slider_with_label(master, text, from_, to_, default_value):
    frame = ttk.Frame(master)
    frame.pack(fill='x', padx=5, pady=5)

    label = ttk.Label(frame, text=text)
    label.pack(side='left')

    slider = ttk.Scale(frame, from_=from_, to=to_, orient='horizontal', command=lambda value: update_simulation())
    slider.set(default_value)  # 使用參數指定的預設值
    slider.pack(side='left', expand=True, fill='x')

    value_label = ttk.Label(frame, text=f"{default_value:.2f}")
    value_label.pack(side='left', padx=5)

    # 更新滑動條值變化時的標籤顯示
    slider.bind("<Motion>", lambda event: value_label.config(text=f"{slider.get():.2f}"))

    return slider, value_label

def safe_close():
    if hasattr(update_simulation, "after_id") and update_simulation.after_id:
        root.after_cancel(update_simulation.after_id)
    root.destroy()

def load_wavelength_intensity_data():
    global target_intensities
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        # 讀取檔案
        wavelengths, target_intensities = np.loadtxt(file_path, unpack=True)

# 創建主窗口
root = tk.Tk()
root.title("折射率調整模擬")


n0_slider, n0_value_label = create_slider_with_label(root, "n0:", 1, 3, 1)
n1_slider, n1_value_label = create_slider_with_label(root, "n1:", 1.3, 2.5, 1.8)
n2_slider, n2_value_label = create_slider_with_label(root, "n2:", 1, 5, 3.4)
d_slider, d_value_label = create_slider_with_label(root, "薄膜厚度 d (nm):", 1000, 15000, 8000)


# theta1 輸入框
theta1_frame = ttk.Frame(root)
theta1_frame.pack(fill='x', padx=5, pady=5)

theta1_label = ttk.Label(theta1_frame, text="theta1 (度):")
theta1_label.pack(side='left')

theta1_entry = ttk.Entry(theta1_frame)
theta1_entry.insert(0, "0")  # 預設值為0
theta1_entry.pack(side='left', expand=True, fill='x')

# 用於展示結果的變量和標籤
result_var = tk.StringVar()
result_label = ttk.Label(root, textvariable=result_var)
result_label.pack()

# Frame for buttons
buttons_frame = ttk.Frame(root)
buttons_frame.pack(pady=20)

# Add Calculate button
calculate_btn = ttk.Button(buttons_frame, text="Calculate", command=update_fig2_only)
calculate_btn.pack(side='left', padx=5)

# Add Export button
export_btn = ttk.Button(buttons_frame, text="Export Data", command=export_data)
export_btn.pack(side='left', padx=5)

# 在buttons_frame中新增一個按鈕用於加載波長和強度數據
load_wavelength_intensity_btn = ttk.Button(buttons_frame, text="Load Data", command=load_wavelength_intensity_data)
load_wavelength_intensity_btn.pack(side='left', padx=5)

# Add Close button
close_btn = ttk.Button(buttons_frame, text="Close", command=safe_close)
close_btn.pack(side='left', padx=5)

# Create a figure and a canvas to display the plot
fig = plt.figure(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)  # Creating a canvas widget in Tkinter
plot_widget = canvas.get_tk_widget()
plot_widget.pack(fill=tk.BOTH, expand=True)

# 在原有的圖形和畫布創建代碼下方添加
fig2 = plt.figure(figsize=(5, 4))
canvas2 = FigureCanvasTkAgg(fig2, master=root)  # 為第二個圖形創建畫布
plot_widget2 = canvas2.get_tk_widget()
plot_widget2.pack(fill=tk.BOTH, expand=True)

# Initialize the GUI with no plot
plt.title('Simulation Results will be displayed here')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Simulation Result')
canvas.draw()

# 初始化模擬結果
update_simulation()

# 運行主循環
root.mainloop()
