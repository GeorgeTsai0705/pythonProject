import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np


class DataVisualizer:
    def __init__(self, root):
        # 初始化設定
        self.root = root
        self.fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  # 將 Figure 放入 Tkinter 的 Canvas 中
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, root)  # 添加 Matplotlib 的導航工具欄

        self.title = [None]
        self.blue_line = [None]
        self.data = [None]
        self.original_data = [None]

        self.ymin_var = tk.StringVar(value='0')
        self.ymax_var = tk.StringVar(value='255')
        self.blue_line_var = tk.BooleanVar()
        self.subtract_var = tk.BooleanVar()

        self.ymin_var.trace("w", self.update_ylim)
        self.ymax_var.trace("w", self.update_ylim)
        self.blue_line_var.trace('w', self.draw_blue_line)
        self.subtract_var.trace('w', self.update_data)

        self.frame1 = tk.Frame(root)
        self.frame2 = tk.Frame(root)

        self.create_widgets()

    def create_widgets(self):
        # 建立並配置視窗元件
        self.frame1.pack()
        self.frame2.pack()

        # 在 Frame1 中添加 Y 軸範圍的 Label、Entry 和藍線的 Checkbutton
        self._create_widget(self.frame1, tk.Label, text="Y min:").pack(side=tk.LEFT)
        self._create_widget(self.frame1, tk.Entry, textvariable=self.ymin_var, width=5).pack(side=tk.LEFT)

        self._create_widget(self.frame1, tk.Label, text="Y max:").pack(side=tk.LEFT)
        self._create_widget(self.frame1, tk.Entry, textvariable=self.ymax_var, width=5).pack(side=tk.LEFT)

        self._create_widget(self.frame1, tk.Checkbutton, text="RefLine", variable=self.blue_line_var).pack(side=tk.LEFT)
        self._create_widget(self.frame1, tk.Checkbutton, text="Subtract", variable=self.subtract_var).pack(side=tk.LEFT)

        # 在 Frame2 中添加 "Open File" 和 "Quit" 兩個按鈕
        self._create_widget(self.frame2, tk.Button, text="Open File", command=self.open_file).pack(side=tk.LEFT)
        self._create_widget(self.frame2, tk.Button, text="Quit", command=self.root.quit).pack(side=tk.LEFT)

    @staticmethod
    def _create_widget(master, widget_class, **kwargs):
        # 建立 widget
        return widget_class(master, **kwargs)

    def open_file(self):
        # 打開文件並讀取數據
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        self.original_data[0] = np.loadtxt(filename, dtype=float)
        self.data[0] = self.original_data[0].copy()
        self.title[0] = os.path.basename(filename)[:11].ljust(11, '*')
        self.update_figure()

    def update_data(self, *args):
        # 更新數據
        if self.subtract_var.get() and self.original_data[0] is not None:
            mean = np.mean(self.original_data[0][:50])
            self.data[0] = self.original_data[0] - mean
        else:
            self.data[0] = self.original_data[0].copy()
        self.update_figure()

    def update_figure(self):
        # 更新畫布
        self.ax.clear()
        if self.data[0] is not None:
            self.ax.plot(self.data[0], 'r-')
            self.ax.set_xlabel('Index')
            self.ax.set_ylabel('Value')
            if self.title[0] is not None:
                self.ax.set_title(self.title[0])
            self.draw_blue_line()

    def update_ylim(self, *args):
        # 更新 Y 軸範圍
        try:
            ymin = float(self.ymin_var.get())
            ymax = float(self.ymax_var.get())
            self.ax.set_ylim([ymin, ymax])
            self.canvas.draw()
        except ValueError:
            pass

    def draw_blue_line(self, *args):
        # 繪製藍線
        if self.blue_line[0] is not None:
            self.blue_line[0].remove()
        if self.blue_line_var.get():
            self.blue_line[0] = self.ax.plot([0, 1280], [0, 0], 'b-')[0]
        else:
            self.blue_line[0] = None
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = DataVisualizer(root)
    root.mainloop()
