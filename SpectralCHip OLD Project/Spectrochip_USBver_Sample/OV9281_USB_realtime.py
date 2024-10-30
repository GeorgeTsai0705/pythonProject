import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image, ImageTk
import cv2
import time
import threading


class VideoAnalyzer(tk.Tk):
    """Pixel Setting for OV9281"""
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720

    def __init__(self):
        super().__init__()
        self.title("Real-Time Video Analyzer")

        # Initialize num_frames
        self.num_frames = 1

        # Load coefficients from file
        self.load_coefficients_from_file("parameters.txt")

        # Initial setup
        self.cap = None
        """Temp Coefficients for Demo"""
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

        # For counting update frequency
        self.last_update_time = None  # 用於追蹤上一次更新的時間
        self.update_rate = 0  # 更新速率（每秒更新次數）

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
        self.entry_start_row.insert(0, f'{self.coefficients[3]}')

        # Label and Entry for end row
        self.lbl_end_row = tk.Label(self.row_frame, text="End Row:")
        self.lbl_end_row.grid(row=0, column=2, padx=5)
        self.entry_end_row = tk.Entry(self.row_frame, width=10)
        self.entry_end_row.grid(row=0, column=3, padx=5)
        self.entry_end_row.insert(0, f'{self.coefficients[3]+20}')

        # Buttons frame
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=10)

        # Button for monitoring
        self.btn_monitor = tk.Button(self.buttons_frame, text="Monitor", command=self.start_monitoring)
        self.btn_monitor.pack(side=tk.LEFT, padx=5)

        # Button for plotting
        self.btn_plot = tk.Button(self.buttons_frame, text="Plot", command=self.plot_graph)
        self.btn_plot.pack(side=tk.LEFT, padx=5)

        # Button for saving spectrum
        self.btn_save_spectrum = tk.Button(self.buttons_frame, text="Save Spectrum", command=self.save_spectrum)
        self.btn_save_spectrum.pack(side=tk.LEFT, padx=5)

        # Button for updating coefficients
        self.btn_update_coeffs = tk.Button(self.buttons_frame, text="Setting",
                                           command=self.open_coefficients_window)
        self.btn_update_coeffs.pack(side=tk.LEFT, padx=5)

        # Quit button
        self.btn_quit = tk.Button(self.buttons_frame, text="Quit", command=self.on_closing)
        self.btn_quit.pack(side=tk.LEFT, padx=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_coefficients_from_file(self, filepath):
        """Load coefficients from a file formatted as a dictionary."""
        try:
            with open(filepath, 'r') as file:
                content = file.read()
                # Using eval to convert string to dictionary, with safety check
                if content.startswith('{') and content.endswith('}'):
                    params = eval(content, {'__builtins__': None}, {})
                    self.coefficients = [params.get('a0'), params.get('a1'), params.get('a2'), params.get('ROI')]
                    print(f"Loaded coefficients: {self.coefficients}")
                else:
                    print("Invalid file format.")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except SyntaxError:
            print("Error parsing the file. Invalid syntax for dictionary.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def find_available_cameras(self):
        """Identify available cameras."""
        available_cameras = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(f"USB {index}")
                cap.release()
                index += 1
            else:
                break
        return available_cameras

    def on_camera_selection(self, _):
        """Handle camera selection and start capturing asynchronously."""
        threading.Thread(target=self.async_camera_selection).start()

    def async_camera_selection(self):
        """Async handling of camera selection."""
        camera_index = int(self.cmb_cameras.get().split()[-1])

        if self.cap is not None:
            self.cap.release()
            time.sleep(0.5)

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print("CamOpenFail")
        """Pixel Setting for OV9281"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.DEFAULT_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.DEFAULT_HEIGHT)
        self.capture_frame()

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

    def init_plot(self):
        self.line.set_data([], [])
        return self.line,

    def open_coefficients_window(self):
        """Open a new window to update coefficients and set number of frames for averaging."""
        self.coeff_window = tk.Toplevel(self)
        self.coeff_window.title("Update Settings")
        self.coeff_window.geometry("300x150")

        # Entry widgets for coefficients
        tk.Label(self.coeff_window, text="a2:").grid(row=0, column=0)
        self.a2_entry = tk.Entry(self.coeff_window,width=10)
        self.a2_entry.grid(row=0, column=1, pady=(10, 5))
        self.a2_entry.insert(0, str(self.coefficients[0]))

        tk.Label(self.coeff_window, text="a1:").grid(row=1, column=0)
        self.a1_entry = tk.Entry(self.coeff_window,width=10)
        self.a1_entry.grid(row=1, column=1, pady=5)
        self.a1_entry.insert(0, str(self.coefficients[1]))

        tk.Label(self.coeff_window, text="a0:").grid(row=2, column=0)
        self.a0_entry = tk.Entry(self.coeff_window,width=10)
        self.a0_entry.grid(row=2, column=1, pady=5)
        self.a0_entry.insert(0, str(self.coefficients[2]))

        # Dropdown for setting number of frames
        tk.Label(self.coeff_window, text="Number of Frames for Averaging:").grid(row=4, column=0)
        self.num_frames_var = tk.StringVar(value=str(self.num_frames))
        self.num_frames_dropdown = ttk.Combobox(self.coeff_window, textvariable=self.num_frames_var, values=[1, 5, 10], width=5)
        self.num_frames_dropdown.grid(row=4, column=1)
        self.num_frames_dropdown.bind("<<ComboboxSelected>>", self.update_num_frames)

        # Update button
        tk.Button(self.coeff_window, text="Update Settings", command=self.update_settings).grid(row=5, columnspan=2)

    def update_settings(self):
        """Update the coefficients with user input."""
        try:
            a0 = float(self.a0_entry.get())
            a1 = float(self.a1_entry.get())
            a2 = float(self.a2_entry.get())
            ROI = int(self.entry_start_row.get())
            self.coefficients = [a0, a1, a2, ROI]

            # Update num_frames
            self.num_frames = int(self.num_frames_var.get())

            print(f"Updated coefficients: {self.coefficients}")
            print(f"Updated number of frames for averaging: {self.num_frames}")
            self.save_coefficients_to_file("parameters.txt")

            self.coeff_window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values.")

    def save_coefficients_to_file(self, filepath):
        """Save the current coefficients to the specified file."""
        try:
            with open(filepath, 'w') as file:
                # Assuming the file format is a simple dictionary
                # Adjust the format as necessary based on the actual file format
                file_content = f"{{'a2': {self.coefficients[0]},\n 'a1': {self.coefficients[1]},\n 'a0': {self.coefficients[2]},\n 'ROI': {self.coefficients[3]}}}"
                file.write(file_content)
            messagebox.showinfo("Success", "Coefficients updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save coefficients: {e}")

    def update_num_frames(self, event):
        """Update num_frames based on the dropdown selection."""
        self.num_frames = int(self.num_frames_var.get())

    def capture_frame(self):
        """Capture video frames and display in UI."""
        if not (self.cap and self.cap.isOpened()):
            return None

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

    def calculate_mean_values(self, frame, start_row, end_row):
        """Calculate mean values from selected rows of a frame."""
        if frame is None or start_row >= end_row:
            return None
        selected_rows = frame[start_row:end_row + 1]
        return np.mean(selected_rows, axis=(0, 2))  # Assuming grayscale for simplicity

    def prepare_plot_data(self, mean_values):
        """Prepare plot data from mean values."""
        if mean_values is None:
            return None, None
        indexes = np.arange(len(mean_values))
        wavelengths = self.coefficients[2] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[0]
        return wavelengths, mean_values

    def update_image_label(self, img):
        """Update image in the tkinter UI."""
        # Desired display size (e.g., 640x480)
        desired_width = 640
        desired_height = 480

        # Resize the image for display but not for analysis
        img = img.resize((desired_width, desired_height), Image.ANTIALIAS)

        self.tk_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_image)

    def start_monitoring(self):
        """Start monitoring and updating the plot in real-time."""
        self.last_update_time = None
        self.update_rate = 0
        self.monitoring_figure, self.ax = plt.subplots()
        self.ax.set_xlim([300, 1000])  # Set x-axis limits
        self.ax.set_ylim([0, 255])  # Set y-axis limits
        self.ax.set_title(f"Real-time Spectrum")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Average Value")
        self.line, = self.ax.plot([], [])
        self.ani = animation.FuncAnimation(self.monitoring_figure, self.update_monitoring_plot,
                                           init_func=self.init_plot, blit=True, interval=200)
        plt.show()

    def update_monitoring_plot(self, _):
        """Update the monitoring plot with the latest data, averaging over a specified number of frames."""
        if not self.original_image:
            return self.line,

        # Initialize accumulation and frame count if not already done
        if not hasattr(self, 'accumulated_means') or not hasattr(self, 'frame_count'):
            self.accumulated_means = np.zeros(self.original_image.width)  # Assuming grayscale for simplicity
            self.frame_count = 0

        # Get the current rows for analysis
        start_row, end_row = self.get_row_range()
        img_data = np.array(self.original_image)
        if 0 <= start_row < img_data.shape[0] and 0 <= end_row < img_data.shape[0] and start_row <= end_row:
            selected_rows = img_data[start_row:end_row + 1]
            mean_values = np.mean(selected_rows, axis=(0, 2))  # Assuming grayscale for simplicity

            # Accumulate the mean values
            self.accumulated_means += mean_values
            self.frame_count += 1

        # Check if enough frames have been accumulated
        if self.frame_count >= self.num_frames:
            # Calculate the average of the accumulated mean values
            average_means = self.accumulated_means / self.frame_count

            # Convert pixel indexes to wavelengths for plotting
            indexes = np.arange(len(average_means))
            wavelengths = self.coefficients[2] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[0]

            # Plotting
            self.line.set_data(wavelengths, average_means)
            self.ax.relim()  # Recompute the ax.dataLim
            self.ax.autoscale_view()  # Update ax.viewLim using the new dataLim

            # Reset for the next set of frames
            self.accumulated_means = np.zeros(self.original_image.width)  # Reset accumulation
            self.frame_count = 0  # Reset frame count

        return self.line,

    def plot_graph(self):
        """Plot the averaged spectrum graph based on selected rows from 10 captured frames."""
        start_row, end_row = self.get_row_range()
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Error", "Camera is not opened!")
            return

        # Initialize an array to accumulate the mean values
        accumulated_means = None
        num_frames =self.num_frames

        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from camera!")
                return

            # Check for valid row values
            if 0 <= start_row < frame.shape[0] and 0 <= end_row < frame.shape[0] and start_row <= end_row:
                selected_rows = frame[start_row:end_row + 1]
                mean_values = np.mean(selected_rows, axis=(0, 2))
                if accumulated_means is None:
                    accumulated_means = mean_values
                else:
                    accumulated_means += mean_values
            else:
                messagebox.showerror("Error", "Invalid row values!")
                return
        # Calculate the average of the accumulated mean values
        average_means = accumulated_means / num_frames

        # Convert pixel indexes to wavelengths
        indexes = np.arange(frame.shape[1])
        wavelengths = self.coefficients[2] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[0]

        # Plotting the averaged spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(wavelengths, average_means)
        plt.title(f"Averaged values of selected rows (Start Row: {start_row}, End Row: {end_row})")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Average Value")
        plt.ylim(300, 1000)
        plt.ylim(0, 255)
        plt.show()

    def save_spectrum(self):
        """Save the averaged spectrum data from 10 captured frames."""
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Error", "Camera is not opened!")
            return

        start_row, end_row = self.get_row_range()
        accumulated_means = None
        num_frames = self.num_frames

        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from camera!")
                return

            if 0 <= start_row < frame.shape[0] and 0 <= end_row < frame.shape[0] and start_row <= end_row:
                selected_rows = frame[start_row:end_row + 1]
                mean_values = np.mean(selected_rows, axis=(0, 2))
                if accumulated_means is None:
                    accumulated_means = mean_values
                else:
                    accumulated_means += mean_values
            else:
                messagebox.showerror("Error", "Invalid row values!")
                return

        average_means = accumulated_means / num_frames

        # Convert pixel indexes to wavelengths
        indexes = np.arange(frame.shape[1])
        wavelengths = self.coefficients[2] * indexes ** 2 + self.coefficients[1] * indexes + self.coefficients[0]

        # Ask for file location to save
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not save_path:
            return

        # Write the averaged spectrum data to file
        with open(save_path, 'w') as f:
            f.write("Wavelength(nm)\tIntensity\n")
            for w, i in zip(wavelengths, average_means):
                f.write(f"{format(w, '.3f')}\t{format(i, '.3f')}\n")

        print(f"Spectrum saved to {save_path}")

    def on_closing(self):
        """Handle app closing event."""
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == '__main__':
    app = VideoAnalyzer()
    app.mainloop()
