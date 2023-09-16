import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_csv_and_plot():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not filepath:
        return

    # 讀取CSV
    df = pd.read_csv(filepath, usecols=[0, 1], header=0)
    df.columns = ['X', 'Z']

    # 繪製圖表
    fig, ax = plt.subplots()
    ax.plot(df["X"], df["Z"])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Plot of X vs Z')

    # 顯示在tkinter視窗上
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, padx=10, pady=10)
    canvas.draw()

def exit_program():
    window.destroy()

window = tk.Tk()
window.title("CSV Plotter")

# 載入CSV按鈕
load_button = tk.Button(window, text="Load CSV", command=load_csv_and_plot)
load_button.grid(row=0, column=0, padx=10, pady=10)

# 退出按鈕
exit_button = tk.Button(window, text="Exit", command=exit_program)
exit_button.grid(row=0, column=1, padx=10, pady=10)

window.mainloop()
