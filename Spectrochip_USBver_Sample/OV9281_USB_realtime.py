import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import cv2
import time
import threading


class VideoAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Video Analyzer")

        # Initial setup
        self.cap = None
        self.coefficients = [0.00002946, 0.6312, 284.4]
        self.original_image = None
        self.mean_values = None

        # Populate and display available cameras
        self.available_cameras = self.find_available_cameras()
        self.setup_ui_elements()

        # Start capturing
        self.capture_frame()

    def setup_ui_elements(self):
        """Initialize and display the UI components."""

        # Camera selection frame
        self.camera_frame = tk.Frame(self)
        self.camera_frame.pack(pady=10)

        # Label for camera selection
        self.lbl_choose_cam = tk.Label(self.camera_frame, text="Choose Cam:")
        self.lbl_choose_cam.pack(side=tk.LEFT, padx=5)

        # Combobox for camera selection
        self.cmb_cameras = ttk.Combobox(self.camera_frame, values=self.available_cameras)
        self.cmb_cameras.pack(side=tk.LEFT)
        if self.available_cameras:
            self.cmb_cameras.current(0)
        self.cmb_cameras.bind("<<ComboboxSelected>>", self.on_camera_selection)

        # Image display
        self.img_label = tk.Label(self)
        self.img_label.pack(pady=20)

        # Pixel size display
        self.pixel_size_label = tk.Label(self, text="Pixel Size: N/A x N/A")
        self.pixel_size_label.pack(pady=10)

        # Frame for start and end rows
        self.row_frame = tk.Frame(self)
        self.row_frame.pack(pady=10)

        # Label and Entry for start row
        self.lbl_start_row = tk.Label(self.row_frame, text="Start Row:")
        self.lbl_start_row.grid(row=0, column=0, padx=5)
        self.entry_start_row = tk.Entry(self.row_frame, width=10)
        self.entry_start_row.grid(row=0, column=1, padx=5)
        self.entry_start_row.insert(0, '355')

        # Label and Entry for end row
        self.lbl_end_row = tk.Label(self.row_frame, text="End Row:")
        self.lbl_end_row.grid(row=0, column=2, padx=5)
        self.entry_end_row = tk.Entry(self.row_frame, width=10)
        self.entry_end_row.grid(row=0, column=3, padx=5)
        self.entry_end_row.insert(0, '375')

        # Buttons frame
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=10)

        # Button for plotting
        self.btn_plot = tk.Button(self.buttons_frame, text="Plot", command=self.plot_graph)
        self.btn_plot.pack(side=tk.LEFT, padx=5)

        # Button for saving spectrum
        self.btn_save_spectrum = tk.Button(self.buttons_frame, text="Save Spectrum", command=self.save_spectrum)
        self.btn_save_spectrum.pack(side=tk.LEFT, padx=5)


        # Quit button
        self.btn_quit = tk.Button(self.buttons_frame, text="Quit", command=self.on_closing)
        self.btn_quit.pack(side=tk.LEFT, padx=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def find_available_cameras(self):
        """Identify available cameras."""
        available_cameras = []
        for i in range(10):  # Check indices 0 to 9
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"USB {i}")
                cap.release()
        return available_cameras

    def on_camera_selection(self, _):
        """Handle camera selection and start capturing asynchronously."""
        threading.Thread(target=self.async_camera_selection).start()

    def async_camera_selection(self):
        """Async handling of camera selection."""
        camera_index = int(self.cmb_cameras.get().split()[-1])

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.5)

        self.cap = cv2.VideoCapture(camera_index)
        """Pixel Setting for OV9281"""
        desired_width = 1280
        desired_height = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        self.capture_frame()

    def on_closing(self):
        """Handle app closing event."""
        if self.cap:
            self.cap.release()
        self.destroy()

    def get_row_range(self):
        """Get start and end row values."""
        start_row, end_row = 0, 0  # Default fallback values
        try:
            start_row = int(self.entry_start_row.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for Start Row.")
            self.entry_start_row.delete(0, tk.END)
            self.entry_start_row.insert(0, '0')

        try:
            end_row = int(self.entry_end_row.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for End Row.")
            self.entry_end_row.delete(0, tk.END)
            self.entry_end_row.insert(0, '0')

        return start_row, end_row

    def capture_frame(self):
        """Capture video frames and display in UI."""
        if not (self.cap and self.cap.isOpened()):
            return

        ret, frame = self.cap.read()
        if ret:
            start_row, end_row = self.get_row_range()
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (0, start_row), (frame.shape[1] - 1, end_row), (0, 0, 255), 1)
            self.original_image = Image.fromarray(frame)
            self.update_image_label(Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)))
            self.pixel_size_label.config(text=f"Pixel Size: {frame.shape[1]} x {frame.shape[0]}")


        if self.winfo_exists():
            self.after(10, self.capture_frame)

    def update_image_label(self, img):
        """Update image in the tkinter UI."""
        # Desired display size (e.g., 640x480)
        desired_width = 640
        desired_height = 480

        # Resize the image for display but not for analysis
        img = img.resize((desired_width, desired_height), Image.ANTIALIAS)

        self.tk_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_image)

    def plot_graph(self):
        """Plot the spectrum graph based on selected rows."""
        start_row, end_row = self.get_row_range()
        if not self.original_image:
            return

        img_data = np.array(self.original_image)
        if 0 <= start_row < img_data.shape[0] and 0 <= end_row < img_data.shape[0] and start_row <= end_row:
            indexes = np.arange(self.original_image.width)
            wavelengths = self.coefficients[0] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[2]
            selected_rows = img_data[start_row:end_row + 1]
            self.mean_values = np.mean(selected_rows, axis=(0, 2))
            plt.figure(figsize=(10, 5))
            plt.plot(wavelengths, self.mean_values)
            plt.title(f"Average values of selected rows (Start Row: {start_row}, End Row: {end_row})")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Average Value")
            plt.show()
        else:
            messagebox.showerror("Error", "Invalid row values!")

    def save_spectrum(self):
        """Save the spectrum data."""
        if self.mean_values is None or self.original_image is None:
            messagebox.showerror("Error", "Please plot the spectrum first!")
            return

        indexes = np.arange(self.original_image.width)
        wavelengths = self.coefficients[0] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[2]

        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not save_path:
            return

        with open(save_path, 'w') as f:
            f.write("Wavelength(nm)\tIntensity\n")
            for w, i in zip(wavelengths, self.mean_values):
                f.write(f"{format(w, '.2f')}\t{format(i, '.2f')}\n")
        print(f"Spectrum saved to {save_path}")

    def on_closing(self):
        """Handle app closing event."""
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == '__main__':
    app = VideoAnalyzer()
    app.mainloop()
