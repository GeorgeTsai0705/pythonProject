import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Toplevel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import savgol_filter
import mplcursors  # Import mplcursors for the interactive cursor


class SampleDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("1D Data Viewer")

        # Create a frame for the controls
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Load Data button
        self.load_button = tk.Button(control_frame, text="Load Data", command=self.load_data)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Create "Further Analysis" button
        self.analysis_button = tk.Button(control_frame, text="Further Analysis", command=self.further_analysis)
        self.analysis_button.pack(side=tk.LEFT, padx=5)

        # Create "Close" button
        self.close_button = tk.Button(control_frame, text="Close", command=self.root.quit)
        self.close_button.pack(side=tk.LEFT, padx=5)

        # Sample index label and scrollbar
        self.index_label = tk.Label(control_frame, text="Sample Index:")
        self.index_label.pack(side=tk.LEFT, padx=5)

        self.index_scale = tk.Scale(control_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_plot)
        self.index_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Create a frame for the plot
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.data = None  # To store the loaded data
        self.fig, self.ax = plt.subplots(figsize=(8, 4))  # Initialize the figure and axis
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty plot with fixed Y-axis
        self.ax.set_ylim(0, 1)  # Set initial Y-axis limits; adjust as needed
        self.ax.set_title("Sample Data at Index 0")
        self.ax.set_xlabel("Wavelength")
        self.ax.set_ylabel("Intensity")
        self.canvas.draw()

        # Calculate the wavelength using the given formula
        self.wavelengths = self.calculate_wavelengths()

    def calculate_wavelengths(self):
        a0 = 295.853
        a1 = 0.63
        a2 = 4.386e-5
        x_wavelength = [a0 + a1 * x + a2 * x * x for x in range(1280)]
        return x_wavelength

    def interpolate_data(self, wavelengths, data):
        # Define the interpolation function
        f = interp1d(wavelengths, data, kind='linear')

        # Define new x-axis points at every 0.5 units
        new_wavelengths = np.arange(min(wavelengths), max(wavelengths), 0.5)

        # Interpolate data at the new wavelength points
        new_data = f(new_wavelengths)

        return new_wavelengths, new_data

    def load_data(self):
        # Open file dialog to select text file
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                # Preprocess the file to remove commas and load data into pandas DataFrame
                with open(file_path, 'r') as file:
                    data = file.read().replace(',', ' ')

                # Convert the cleaned data into a pandas DataFrame
                from io import StringIO
                self.data = pd.read_csv(StringIO(data), delim_whitespace=True, header=None)

                messagebox.showinfo("Success", "Data loaded successfully!")

                # Update the scrollbar range based on the number of samples
                self.index_scale.config(to=len(self.data) - 1)

                # Adjust Y-axis based on the data range
                self.ax.set_ylim(self.data.min().min(), self.data.max().max())

                # Show initial plot for index 0
                self.update_plot(0)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")

    def update_plot(self, index):
        if self.data is not None:
            try:
                # Convert index to integer (from string)
                index = int(index)

                # Interpolate the data
                new_wavelengths, new_data = self.interpolate_data(self.wavelengths, self.data.iloc[index])

                # Clear the existing plot
                self.ax.clear()

                # Plot the interpolated data
                line, = self.ax.plot(new_wavelengths, new_data, marker='o', markersize=3, linewidth=2)
                self.ax.set_title(f"Sample Data at Index {index}")
                self.ax.set_xlabel("Wavelength")
                self.ax.set_ylabel("Intensity")
                self.ax.set_xlim(350,800)

                # Fix Y-axis limits to the initial calculated range
                self.ax.set_ylim(self.data.min().min(), self.data.max().max())

                # Refresh the canvas to show the updated plot
                self.canvas.draw()

                # Add the interactive cursor
                mplcursors.cursor(line, hover=True)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to plot data: {e}")

    def further_analysis(self):
        # Open a new window for the user to enter a wavelength and reference frame
        analysis_window = Toplevel(self.root)
        analysis_window.title("Further Analysis")

        tk.Label(analysis_window, text="Enter Wavelength:").pack(pady=10)
        wavelength_entry = tk.Entry(analysis_window)
        wavelength_entry.pack(pady=5)

        tk.Label(analysis_window, text="Enter Reference Frame:").pack(pady=10)
        ref_frame_entry = tk.Entry(analysis_window)
        ref_frame_entry.pack(pady=5)

        def analyze_wavelength():
            try:
                wavelength = float(wavelength_entry.get())
                ref_frame = int(ref_frame_entry.get())

                if ref_frame < 0 or ref_frame >= len(self.data):
                    raise ValueError("Reference frame out of range.")

                # Find the closest wavelength in the calculated wavelengths
                closest_wavelength = min(self.wavelengths, key=lambda x: abs(x - wavelength))

                # Get the reference frame's intensity at the closest wavelength
                new_wavelengths, ref_data = self.interpolate_data(self.wavelengths, self.data.iloc[ref_frame])
                ref_intensity = np.interp(closest_wavelength, new_wavelengths, ref_data)

                if ref_intensity == 0:
                    raise ValueError("Reference frame intensity is zero, cannot divide by zero.")

                # Collect and normalize data at the closest wavelength across all samples
                normalized_intensity_values = []
                for i in range(len(self.data)):
                    new_wavelengths, new_data = self.interpolate_data(self.wavelengths, self.data.iloc[i])
                    intensity = np.interp(closest_wavelength, new_wavelengths, new_data)
                    normalized_intensity_values.append(intensity / ref_intensity)

                # Plot the normalized trend in a new window
                self.show_trend(closest_wavelength, normalized_intensity_values)

            except ValueError as ve:
                messagebox.showerror("Invalid Input", f"Error: {ve}")

        tk.Button(analysis_window, text="Analyze", command=analyze_wavelength).pack(pady=10)
        tk.Button(analysis_window, text="Close", command=analysis_window.destroy).pack(pady=5)

    def show_trend(self, wavelength, intensity_values):
        trend_window = Toplevel(self.root)
        trend_window.title(f"Reflectance Trend Analysis at Wavelength {wavelength:.2f}")

        # Apply Savitzky-Golay filter to smooth the intensity values
        window_length = 15  # Window length, must be odd
        polyorder = 3  # Polynomial order
        sg_intensity_values = savgol_filter(intensity_values, window_length, polyorder)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(sg_intensity_values)), sg_intensity_values, marker='o', markersize=3, linewidth=2)
        ax.set_title(f"Reflectance Intensity Trend at Wavelength {wavelength:.2f}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Reflectance Intensity")

        canvas = FigureCanvasTkAgg(fig, master=trend_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        def save_data():
            # Open a file dialog to select where to save the CSV file
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"),
                                                                                         ("All Files", "*.*")])
            if save_path:
                try:
                    # Save the data as a CSV file
                    df = pd.DataFrame({
                        'Sample Index': range(len(sg_intensity_values)),
                        'Reflectance Intensity': sg_intensity_values
                    })
                    df.to_csv(save_path, index=False)
                    messagebox.showinfo("Saved", "Trend analysis data saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save the data: {e}")

        # Add a Save button to the trend window
        save_button = tk.Button(trend_window, text="Save", command=save_data)
        save_button.pack(pady=10)
if __name__ == "__main__":
    root = tk.Tk()
    app = SampleDataViewer(root)
    root.mainloop()
