import os
import pandas as pd
import json
from tkinter import Tk
from tkinter.filedialog import askdirectory
import numpy as np
from scipy.signal import find_peaks


def read_wavelength(file_path):
    """讀取 UI_parameters.txt 中的 Wavelength (WL) 參數"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data.get("Wavelength (WL)", [])
    except Exception as e:
        print(f"Failed to read Wavelength (WL) from {file_path}: {e}")
        return []


def calculate_fwhm(wavelengths, intensities, reference_wavelengths):
    """計算波峰的半高全寬 (FWHM) 並將波峰波長匹配到最近的參考波長"""
    peaks, properties = find_peaks(intensities, width=4, prominence=7.0, height=250)

    fwhm_results = []
    for peak in peaks:
        half_max = intensities[peak] / 2

        # 左側半高
        left_idx = np.where(intensities[:peak] <= half_max)[0]
        if len(left_idx) > 0:
            left_idx = left_idx[-1]
        else:
            left_idx = 0

        # 右側半高
        right_idx = np.where(intensities[peak:] <= half_max)[0]
        if len(right_idx) > 0:
            right_idx = right_idx[0] + peak
        else:
            right_idx = len(intensities) - 1

        # 對應的波長
        left_wl = np.interp(half_max, [intensities[left_idx], intensities[left_idx + 1]],
                            [wavelengths[left_idx], wavelengths[left_idx + 1]])
        right_wl = np.interp(half_max, [intensities[right_idx - 1], intensities[right_idx]],
                             [wavelengths[right_idx - 1], wavelengths[right_idx]])

        fwhm = right_wl - left_wl

        # 匹配最近的參考波長
        peak_wl = wavelengths[peak]
        closest_wl = round(min(reference_wavelengths, key=lambda x: abs(x - peak_wl)),2)

        fwhm_results.append((peak, closest_wl, fwhm))

    return fwhm_results


def load_chip_data(parent_folder):
    """讀取主資料夾中每個晶片資料夾的 spectrum_data.csv 與 UI_parameters.txt，並進行 Pixel -> Wavelength 座標轉換"""
    reference_wavelengths = [404.656, 435.833, 546.074, 578.013, 696.543, 706.722, 727.294, 738.398, 750.764, 763.511,
                             772.395, 794.818, 801.080, 811.531, 826.452, 841.807, 852.144, 866.794, 912.297, 922.450]
    chip_data = {}

    for chip_folder in os.listdir(parent_folder):
        chip_path = os.path.join(parent_folder, chip_folder)

        if os.path.isdir(chip_path):
            spectrum_file = os.path.join(chip_path, 'spectrum_data.csv')
            ui_params_file = os.path.join(chip_path, 'UI_parameters.txt')

            # 檢查並讀取 spectrum_data.csv
            spectrum = None
            if os.path.exists(spectrum_file):
                try:
                    spectrum = pd.read_csv(spectrum_file)
                except Exception as e:
                    print(f"Failed to load spectrum_data.csv for chip {chip_folder}: {e}")

            # 檢查並讀取 UI_parameters.txt 的 Wavelength (WL)
            wavelength = None
            if os.path.exists(ui_params_file):
                wavelength = read_wavelength(ui_params_file)

            # 若 spectrum 與 wavelength 都存在，進行座標轉換與 FWHM 計算
            if spectrum is not None and wavelength:
                try:
                    # 確保波長參數長度正確
                    if len(wavelength) >= 3:
                        WL = wavelength
                        spectrum['Wavelength_nm'] = (
                                WL[0] +
                                WL[1] * spectrum['Pixel'] +
                                WL[2] * spectrum['Pixel'] ** 2
                        )

                        # 計算 FWHM
                        fwhm_results = calculate_fwhm(spectrum['Wavelength_nm'].values, spectrum['Intensity'].values,
                                                      reference_wavelengths)
                        chip_data[chip_folder] = {
                            "spectrum": spectrum,
                            "wavelength": wavelength,
                            "fwhm_results": fwhm_results
                        }

                        # 儲存 fwhm_table.csv
                        fwhm_table = pd.DataFrame(fwhm_results,
                                                  columns=["Peak Position", "Wavelength (nm)", "FWHM (nm)"])
                        output_path = os.path.join(chip_path, 'fwhm_table.csv')
                        fwhm_table.to_csv(output_path, index=False)
                        print(f"FWHM table saved to {output_path}")
                except Exception as e:
                    print(f"Error during wavelength conversion or FWHM calculation for chip {chip_folder}: {e}")

    return chip_data


# 主程式
if __name__ == "__main__":
    # 使用 Tkinter 彈出視窗選擇資料夾
    Tk().withdraw()
    parent_folder_path = askdirectory(title="選擇主資料夾")

    if parent_folder_path:
        print(f"您選擇的資料夾是: {parent_folder_path}")
        chip_data = load_chip_data(parent_folder_path)

        # 輸出每個晶片的資料
        for chip, data in chip_data.items():
            spectrum_shape = data["spectrum"].shape if data["spectrum"] is not None else "No spectrum"
            wavelength_params = data["wavelength"]
            fwhm_results = data.get("fwhm_results", [])
            print(f"Chip: {chip}, Spectrum shape: {spectrum_shape}, Wavelength parameters: {wavelength_params}")
            for peak_pixel, peak_wl, fwhm in fwhm_results:
                print(f"  Peak Position: {peak_pixel}, Wavelength (nm): {peak_wl:.2f}, FWHM (nm): {fwhm:.2f}")
    else:
        print("未選擇任何資料夾。")
