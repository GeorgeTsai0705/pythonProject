import os
import numpy as np
import pandas as pd


def interpolate_data(wavelengths, intensities, start_wavelength=350, end_wavelength=950, interval=1):
    # Interpolate intensity values over a specified range of wavelengths
    integer_wavelengths = np.arange(start_wavelength, end_wavelength + 1, interval)
    interpolated_intensities = np.interp(integer_wavelengths, wavelengths, intensities)
    return pd.DataFrame({'Wavelength': integer_wavelengths, 'Interpolated Intensity': interpolated_intensities})


def recalculate_wavelengths(A0, A1, A2):
    # Calculate new wavelengths using the provided coefficients
    return [A0 + A1 * i + A2 * i ** 2 for i in range(1280)]


def process_directory(directory_path, coeff_table):
    results = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]

                # Lookup in coeff_table
                coeff_row = coeff_table.loc[coeff_table['Filename'] == base_name]
                if not coeff_row.empty:
                    A0, A1, A2 = coeff_row.iloc[0, 1:4].values
                    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['Wavelength', 'Intensity'])
                    intensities = data.iloc[:, 0]  # Directly use the first column as intensities
                    wavelengths = recalculate_wavelengths(A0, A1, A2)
                    interpolated_data = interpolate_data(wavelengths, intensities)
                    results.append((file, interpolated_data))
                else:
                    print(f"Warning: No coefficients found for {file}")
    return results


def save_interpolated_data(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name, data in results:
        base_name, ext = os.path.splitext(file_name)
        output_file = os.path.join(output_dir, f"{base_name}_inter{ext}")
        # Save with a space as the separator
        data.to_csv(output_file, sep=' ', index=False)


# 設定三個資料夾的路徑
data_path = "D:\博士班\Conference\MEMS2025\Spectrum Sample"
folders = ['Good', 'Baseline', 'Misfocus']
output_dir = 'Interpolated_Output'

# Load Coeff_Table.csv with high precision for float columns
coeff_table_path = os.path.join(data_path, "Coeff_Table.csv")
coeff_table = pd.read_csv(coeff_table_path,
                          dtype={'Filename': str, 'A0': np.float64, 'A1': np.float64, 'A2': np.float64})

# 處理每個資料夾中的光譜檔案並將結果保存到同一個資料夾中
for folder in folders:
    results = process_directory(os.path.join(data_path, folder), coeff_table)
    save_interpolated_data(results, os.path.join(data_path, output_dir))

print("All spectra files have been processed, recalculated data saved.")
