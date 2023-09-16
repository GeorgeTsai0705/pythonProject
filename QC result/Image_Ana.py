import tkinter as tk
from tkinter import filedialog
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

        self.filename_label = tk.Label(self, text="", font=("Arial", 12, "bold"))  # label for showing filename
        self.filename_label.pack(pady=10)

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=20)

        self.btn_select_file = tk.Button(self, text="Select BMP File", command=self.select_file)
        self.btn_select_file.pack(pady=20)

        self.lbl_start_row = tk.Label(self, text="Start Row:")
        self.lbl_start_row.pack(pady=5)

        self.entry_start_row = tk.Entry(self)
        self.entry_start_row.pack(pady=5)

        self.lbl_end_row = tk.Label(self, text="End Row:")
        self.lbl_end_row.pack(pady=5)

        self.entry_end_row = tk.Entry(self)
        self.entry_end_row.pack(pady=5)

        self.btn_plot = tk.Button(self, text="Plot", command=self.plot_graph)
        self.btn_plot.pack(pady=20)

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

    def plot_graph(self):
        if not self.original_image:
            print("Please select a BMP file first!")
            return

        try:
            start_row = int(self.entry_start_row.get())
            end_row = int(self.entry_end_row.get())
        except ValueError:
            print("Please enter valid row values!")
            return

        img_data = np.array(self.original_image)

        # Check the validity of row values
        if start_row < 0 or start_row >= img_data.shape[0] or end_row < 0 or end_row >= img_data.shape[
            0] or start_row > end_row:
            print("Invalid row values!")
            return

        # Draw red lines on the image at the start and end rows
        img_with_lines = self.original_image.copy()
        draw = ImageDraw.Draw(img_with_lines)
        draw.line([(0, start_row), (img_with_lines.width, start_row)], fill="red", width=2)
        draw.line([(0, end_row), (img_with_lines.width, end_row)], fill="red", width=2)
        self.update_image_label(img_with_lines)

        # Calculate the mean values for the selected rows
        selected_rows = img_data[start_row:end_row + 1]
        mean_values = np.mean(selected_rows, axis=(0, 2))

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(mean_values)
        title = f"Average values of selected rows (Start Row: {start_row}, End Row: {end_row})"
        plt.title(title)
        plt.xlabel("Column Index")
        plt.ylabel("Average Value")
        plt.show()


if __name__ == '__main__':
    app = BMPAnalyzer()
    app.mainloop()
