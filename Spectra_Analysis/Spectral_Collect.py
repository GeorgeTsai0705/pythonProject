import os
import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
from typing import Optional, Dict, Any


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Quadratic function: a * x^2 + b * x + c.

    Args:
        x (np.ndarray): The independent variable array.
        a (float): Quadratic coefficient.
        b (float): Linear coefficient.
        c (float): Constant term.

    Returns:
        np.ndarray: Evaluated quadratic function values.
    """
    return a * x ** 2 + b * x + c


def safe_load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content if successful; otherwise, None.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def safe_load_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame if successful; otherwise, None.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def analyze_chips(base_folder: str) -> None:
    """
    Analyze chip data from subfolders in the specified base folder.
    For each chip folder, the function loads UI parameters, spectrum data,
    and FWHM data, computes various features (including quadratic peak fits),
    and then outputs a summary CSV.

    Args:
        base_folder (str): Path to the folder containing chip subdirectories.
    """
    chip_summary = []

    # Custom peak wavelengths (modifiable by the user)
    custom_peak_wavelengths = [435.83, 546.07, 763.51, 811.53]

    # Iterate over each subfolder (each representing a chip)
    for chip_id in os.listdir(base_folder):
        chip_folder = os.path.join(base_folder, chip_id)
        if not os.path.isdir(chip_folder):
            continue  # Skip files or non-directory items

        # Define paths for the required files
        ui_parameters_path = os.path.join(chip_folder, "UI_parameters.txt")
        spectrum_data_path = os.path.join(chip_folder, "spectrum_data.csv")
        fwhm_table_path = os.path.join(chip_folder, "fwhm_table.csv")

        # Load UI parameters (JSON format expected)
        ui_parameters = safe_load_json(ui_parameters_path)
        if ui_parameters is None:
            continue  # Skip chip if UI parameters cannot be loaded

        # Load spectrum data and FWHM table
        spectrum_data = safe_load_csv(spectrum_data_path)
        if spectrum_data is None:
            continue

        fwhm_table = safe_load_csv(fwhm_table_path)
        if fwhm_table is None:
            continue

        # --- Wavelength to Pixel Coordinate Transformation ---
        # If the UI parameters provide wavelength calibration coefficients,
        # compute the wavelength for each pixel index.
        if 'Wavelength (WL)' in ui_parameters:
            wl_coeffs = ui_parameters['Wavelength (WL)']
            if isinstance(wl_coeffs, list) and len(wl_coeffs) >= 3:
                x_indices = np.array(spectrum_data.index)
                a, b, c = wl_coeffs[:3]
                spectrum_data['Wavelength_nm'] = a + b * x_indices + c * (x_indices ** 2)

        # --- Initialize Feature Variables ---
        feature1_intensity = None
        feature2_intensity = None
        feature3_intensity = None
        feature4_difference = None

        # --- Compute Features based on Spectrum Data ---
        if 'Wavelength_nm' in spectrum_data.columns:
            # Feature 1: Maximum intensity in the range [459, 530] nm
            filtered_spectrum1 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 459) &
                                               (spectrum_data['Wavelength_nm'] <= 530)]
            if not filtered_spectrum1.empty:
                feature1_intensity = filtered_spectrum1['Intensity'].max()

            # Feature 2: Maximum intensity in the range [600, 668] nm
            filtered_spectrum2 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 600) &
                                               (spectrum_data['Wavelength_nm'] <= 668)]
            if not filtered_spectrum2.empty:
                feature2_intensity = filtered_spectrum2['Intensity'].max()

            # Feature 3: Minimum intensity in the range [777, 792] nm
            filtered_spectrum3 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 777) &
                                               (spectrum_data['Wavelength_nm'] <= 792)]
            if not filtered_spectrum3.empty:
                feature3_intensity = filtered_spectrum3['Intensity'].min()

            # Feature 4: Use quadratic fitting to estimate FWHM differences at custom peaks
            f1_fitted_list = []
            f2_actual_list = []
            for peak_wavelength in custom_peak_wavelengths:
                # Select data within ±3 nm of the custom peak wavelength
                peak_data = spectrum_data[(spectrum_data['Wavelength_nm'] >= peak_wavelength - 3) &
                                          (spectrum_data['Wavelength_nm'] <= peak_wavelength + 3)]
                if peak_data.empty:
                    continue

                # Identify the index of maximum intensity near the peak
                center_index = peak_data['Intensity'].idxmax()
                # Extend the data selection for a robust quadratic fit
                start_index = max(0, center_index - 3)
                end_index = center_index + 4  # end index is exclusive in iloc slicing
                extended_peak_data = spectrum_data.iloc[start_index:end_index]
                x_data = extended_peak_data['Wavelength_nm']
                y_data = extended_peak_data['Intensity']

                try:
                    # Fit the quadratic model to the selected data
                    params, _ = curve_fit(quadratic, x_data, y_data)
                    a_fit, b_fit, c_fit = params

                    # Estimate the peak position using the vertex of the quadratic
                    fitted_peak_position = -b_fit / (2 * a_fit)
                    # Calculate the intensity at the peak and then half of that intensity
                    peak_intensity = quadratic(fitted_peak_position, a_fit, b_fit, c_fit)
                    half_peak_intensity = peak_intensity / 2

                    # Adjust the constant term to determine the half-maximum crossing points
                    c_shifted = c_fit - half_peak_intensity
                    roots = np.roots([a_fit, b_fit, c_shifted])

                    # Only accept real roots (two points) for a valid FWHM estimate
                    if len(roots) == 2 and np.all(np.isreal(roots)):
                        roots = np.real(roots)
                        f1_fitted = abs(roots[1] - roots[0])

                        # Retrieve the actual FWHM from the table for the current peak wavelength
                        row_mask = fwhm_table['Wavelength (nm)'] == peak_wavelength
                        if row_mask.any():
                            f2_actual = fwhm_table.loc[row_mask, 'FWHM (nm)'].values[0]
                            f1_fitted_list.append(f1_fitted)
                            f2_actual_list.append(f2_actual)
                except Exception as e:
                    print(f"Error fitting quadratic for chip {chip_id}, peak {peak_wavelength}: {e}")

            # Calculate the average absolute difference between fitted and actual FWHM values
            if f1_fitted_list and f2_actual_list:
                feature4_difference = np.mean(np.abs(np.array(f1_fitted_list) - np.array(f2_actual_list)))

        # --- Spectrum Summary Statistics ---
        spectrum_max = spectrum_data['Intensity'].max()
        # Avoid division by zero; set to NaN if spectrum_max is zero or None
        if not spectrum_max:
            spectrum_max = np.nan

        # Calculate intensity ratios (relative to the overall maximum)
        feature1_ratio = feature1_intensity / spectrum_max if feature1_intensity is not None and spectrum_max != 0 else None
        feature2_ratio = feature2_intensity / spectrum_max if feature2_intensity is not None and spectrum_max != 0 else None
        feature3_ratio = feature3_intensity / spectrum_max if feature3_intensity is not None and spectrum_max != 0 else None

        # --- Feature 5: Derived Calculation ---
        # Computed from Gain, Exposure Time, and spectrum_max (adjust the formula as needed)
        feature5 = None
        gain = ui_parameters.get('Gain')
        exposure_time = ui_parameters.get('Exposure Time (ms)')
        if gain is not None and exposure_time is not None and spectrum_max and spectrum_max != 0:
            feature5 = ((gain / 32) * (exposure_time / 900)) / (spectrum_max / 4000)

        # --- Feature 6: Ratio in a Specific Wavelength Range ---
        # Maximum intensity in [410, 550] nm divided by the overall spectrum maximum.
        feature6 = None
        if 'Wavelength_nm' in spectrum_data.columns and spectrum_max and spectrum_max != 0:
            filtered_spectrum6 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 410) &
                                               (spectrum_data['Wavelength_nm'] <= 550)]
            if not filtered_spectrum6.empty:
                max_intensity_in_range = filtered_spectrum6['Intensity'].max()
                feature6 = max_intensity_in_range / spectrum_max

        # --- Consolidate Spectrum Statistics ---
        spectrum_stats = {
            'spectrum_max': spectrum_max,
            'feature1_max_intensity': feature1_intensity,
            'feature2_max_intensity': feature2_intensity,
            'feature3_min_intensity': feature3_intensity,
            'feature1_ratio': feature1_ratio,
            'feature2_ratio': feature2_ratio,
            'feature3_ratio': feature3_ratio,
            'feature4_difference': feature4_difference,
            'feature5': feature5,
            'feature6': feature6
        }

        # --- Extract FWHM Statistics for Selected Wavelengths ---
        selected_wavelengths = [404.66, 435.83, 546.07, 578.01, 696.54, 763.51, 811.53, 841.81]
        filtered_fwhm = fwhm_table[fwhm_table['Wavelength (nm)'].isin(selected_wavelengths)]
        fwhm_stats = {
            'fwhm_min': filtered_fwhm['FWHM (nm)'].min() if not filtered_fwhm.empty else None,
            'fwhm_max': filtered_fwhm['FWHM (nm)'].max() if not filtered_fwhm.empty else None,
            'fwhm_mean': filtered_fwhm['FWHM (nm)'].mean() if not filtered_fwhm.empty else None,
            'fwhm_std': filtered_fwhm['FWHM (nm)'].std() if not filtered_fwhm.empty else None
        }

        # --- Extract UI Parameters for the Summary ---
        ui_stats = {
            'ROI': ui_parameters.get('ROI'),
            'Exposure Time (ms)': ui_parameters.get('Exposure Time (ms)'),
            'Gain': ui_parameters.get('Gain'),
            'Baseline Evaluation': ui_parameters.get('Baseline Evaluation'),
            'Wavelength (WL)': ui_parameters.get('Wavelength (WL)')
        }

        # --- Combine All Information for the Current Chip ---
        chip_info = {
            'ChipID': chip_id,
            **ui_stats,
            **spectrum_stats,
            **fwhm_stats
        }
        chip_summary.append(chip_info)

    # --- Create and Save the Summary DataFrame ---
    summary_df = pd.DataFrame(chip_summary)
    summary_output_path = os.path.join(base_folder, "chip_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Chip summary saved to {summary_output_path}")


if __name__ == "__main__":
    # Launch a file dialog to select the folder containing chip data
    Tk().withdraw()  # Hide the main Tk window
    folder_path = askdirectory(title="Select the folder containing chip data")
    if folder_path:
        analyze_chips(folder_path)
    else:
        print("No folder selected.")
