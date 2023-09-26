import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os


class BMPAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BMP Image Analyzer")

        self.filepath = None
        self.original_image = None
        self.mean_values = None

        # 定義一元二次方程式的係數
        self.coefficients = [0.00002946, 0.6312, 284.4]

        self.filename_label = tk.Label(self, text="", font=("Arial", 12, "bold"))  # label for showing filename
        self.filename_label.pack(pady=10)

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=20)

        self.btn_select_file = tk.Button(self, text="Select BMP File", command=self.select_file)
        self.btn_select_file.pack(pady=20)

        self.lbl_start_row = tk.Label(self, text="Start Row:")
        self.lbl_start_row.pack(pady=5)

        self.entry_start_row = tk.Entry(self, width=10)
        self.entry_start_row.pack(pady=5)

        self.lbl_end_row = tk.Label(self, text="End Row:")
        self.lbl_end_row.pack(pady=5)

        self.entry_end_row = tk.Entry(self, width=10)
        self.entry_end_row.pack(pady=5)

        self.btn_plot = tk.Button(self, text="Plot", command=self.plot_graph)
        self.btn_plot.pack(pady=20)

        self.btn_save_spectrum = tk.Button(self, text="Save Spectrum", command=self.save_spectrum)
        self.btn_save_spectrum.pack(pady=20)

    def select_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if self.filepath:
            self.original_image = Image.open(self.filepath)
            self.update_image_label(self.original_image)

            # Extract filename without extension and update the filename label
            base_name = os.path.basename(self.filepath)
            file_name, _ = os.path.splitext(base_name)
            self.filename_label.config(text=file_name)

    def update_image_label(self, img):
        img = img.resize((250, 250))  # You can adjust the size if needed
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_image)

    def save_spectrum(self):
        # Check if start and end rows are entered
        start_row_value = self.entry_start_row.get()
        end_row_value = self.entry_end_row.get()

        if not start_row_value or not end_row_value:
            tk.messagebox.showerror("Error", "Please enter Start Row and End Row first!")
            return

        if self.mean_values is None or self.original_image is None:
            tk.messagebox.showerror("Error", "Please plot the spectrum first!")
            return

        # 計算圖片的 column indexes
        indexes = np.arange(self.original_image.width)

        # 使用一元二次方程式將 indexes 轉換成 wavelengths
        wavelengths = self.coefficients[0] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[2]

        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not save_path:
            return

        with open(save_path, 'w') as f:
            f.write("Wavelength(nm)\tIntensity\n")  # 寫入標題
            for w, i in zip(wavelengths, self.mean_values):
                f.write(f"{format(w, '.2f')}\t{format(i, '.2f')}\n")  # 同時輸出 wavelength 和 intensity
        print(f"Spectrum saved to {save_path}")

    def plot_graph(self):

        if not self.original_image:
            messagebox.showerror("Error", "Please select a BMP file first!")
            return

        try:
            start_row = int(self.entry_start_row.get())
            end_row = int(self.entry_end_row.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid row values!")
            return

        img_data = np.array(self.original_image)

        # Check the validity of row values
        if start_row < 0 or start_row >= img_data.shape[0] or end_row < 0 or end_row >= img_data.shape[0] or start_row > end_row:
            messagebox.showerror("Error", "Invalid row values!")
            return

        indexes = np.arange(self.original_image.width)
        # 使用一元二次方程式將 indexes 轉換成 wavelengths
        wavelengths = self.coefficients[0] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[2]

        # Draw red lines on the image at the start and end rows
        img_with_lines = self.original_image.copy()
        draw = ImageDraw.Draw(img_with_lines)
        draw.line([(0, start_row), (img_with_lines.width, start_row)], fill="red", width=2)
        draw.line([(0, end_row), (img_with_lines.width, end_row)], fill="red", width=2)
        self.update_image_label(img_with_lines)

        # Calculate the mean values for the selected rows
        selected_rows = img_data[start_row:end_row + 1]
        mean_values = np.mean(selected_rows, axis=(0, 2))

        # Store the mean_values to self after plotting
        self.mean_values = mean_values

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(wavelengths, mean_values)
        title = f"Average values of selected rows (Start Row: {start_row}, End Row: {end_row})"
        plt.title(title)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Average Value")
        plt.show()


if __name__ == '__main__':
    app = BMPAnalyzer()
    app.mainloop()
