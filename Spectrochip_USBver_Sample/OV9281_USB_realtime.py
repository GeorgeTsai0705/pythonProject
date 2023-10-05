import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import cv2
import time  # Import time for sleep
import threading  # Import threading for async operations

class VideoAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Video Analyzer")

        self.cap = None  # Initialize the cap object as None

        self.mean_values = None
        self.coefficients = [0.00002946, 0.6312, 284.4]

        self.available_cameras = self.find_available_cameras()
        self.cmb_cameras = ttk.Combobox(self, values=self.available_cameras)
        self.cmb_cameras.pack(pady=10)
        if self.available_cameras:
            self.cmb_cameras.current(0)  # Set default selection to the first available camera
        self.cmb_cameras.bind("<<ComboboxSelected>>", self.on_camera_selection)

        self.mean_values = None
        self.coefficients = [0.00002946, 0.6312, 284.4]

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=20)

        self.lbl_start_row = tk.Label(self, text="Start Row:")
        self.lbl_start_row.pack(pady=5)

        self.entry_start_row = tk.Entry(self, width=10)
        self.entry_start_row.pack(pady=5)
        self.entry_start_row.insert(0, '230')

        self.lbl_end_row = tk.Label(self, text="End Row:")
        self.lbl_end_row.pack(pady=5)

        self.entry_end_row = tk.Entry(self, width=10)
        self.entry_end_row.pack(pady=5)
        self.entry_end_row.insert(0, '250')

        self.btn_plot = tk.Button(self, text="Plot", command=self.plot_graph)
        self.btn_plot.pack(pady=20)

        self.btn_save_spectrum = tk.Button(self, text="Save Spectrum", command=self.save_spectrum)
        self.btn_save_spectrum.pack(pady=20)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.capture_frame()

    def find_available_cameras(self):
        available_cameras = []
        for i in range(10):  # Check indices 0 to 9
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"USB {i}")
                cap.release()  # Release the camera immediately after checking
        return available_cameras

    def on_camera_selection(self, event):
        # Run the camera selection in a separate thread to avoid UI freezing
        threading.Thread(target=self.async_camera_selection).start()

    def async_camera_selection(self):
        camera_index = int(self.cmb_cameras.get().split()[-1])  # Extract camera index from selection

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()  # Release the previous camera if it's opened
            time.sleep(0.5)  # Small delay to ensure previous camera is released

        self.cap = cv2.VideoCapture(camera_index)
        self.capture_frame()  # Start capturing from the newly selected camera

    def get_row_range(self):
        try:
            start_row = int(self.entry_start_row.get())
            end_row = int(self.entry_end_row.get())
            return start_row, end_row
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter valid numbers for Start Row and End Row.")
            # Use default values if row values are not valid
            return 230, 250

    def capture_frame(self):
        if self.cap is None or not self.cap.isOpened():  # Check if a camera is selected and opened
            return
        ret, frame = self.cap.read()  # Read a frame from the video
        if ret:
            row_range = self.get_row_range()
            if row_range is not None:
                start_row, end_row = row_range
            else:
                # Use default values if row values are not valid
                start_row, end_row = 0, frame.shape[0] - 1

            # Create a copy of the frame for drawing
            frame_copy = frame.copy()

            # Draw a red rectangle for the ROI on the copy
            cv2.rectangle(frame_copy, (0, start_row), (frame.shape[1] - 1, end_row), (0, 0, 255), 1)

            self.original_image = Image.fromarray(frame)
            self.update_image_label(Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)))

        if self.winfo_exists():  # Check if the window still exists
            self.after(10, self.capture_frame)  # If the window still exists, capture a frame every 10 milliseconds

    def update_image_label(self, img):
        img = img.resize((640, 480))  # You can adjust the size if needed
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_image)

    def save_spectrum(self):
        row_range = self.get_row_range()
        if row_range is None:
            return

        if self.mean_values is None or self.original_image is None:
            tk.messagebox.showerror("Error", "Please plot the spectrum first!")
            return

        # Calculate the column indexes of the image
        indexes = np.arange(self.original_image.width)

        # Convert indexes to wavelengths using the quadratic equation
        wavelengths = self.coefficients[0] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[2]

        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not save_path:
            return

        with open(save_path, 'w') as f:
            f.write("Wavelength(nm)\tIntensity\n")  # Write the header
            for w, i in zip(wavelengths, self.mean_values):
                f.write(f"{format(w, '.2f')}\t{format(i, '.2f')}\n")  # Output wavelength and intensity simultaneously
        print(f"Spectrum saved to {save_path}")

    def plot_graph(self):
        row_range = self.get_row_range()
        if row_range is None or not self.original_image:
            return

        start_row, end_row = row_range
        img_data = np.array(self.original_image)

        # Check the validity of row values
        if 0 <= start_row < img_data.shape[0] and 0 <= end_row < img_data.shape[0] and start_row <= end_row:
            indexes = np.arange(self.original_image.width)
            # Convert indexes to wavelengths using the quadratic equation
            wavelengths = self.coefficients[0] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[2]

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
        else:
            tk.messagebox.showerror("Error", "Invalid row values!")

    def on_closing(self):
        self.cap.release()  # Release the video capture object
        self.destroy()  # Close the tkinter window

if __name__ == '__main__':
    app = VideoAnalyzer()
    app.mainloop()
