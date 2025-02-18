import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# 載入光譜資料
def load_spectrum(file_path):
    """
    從文件中讀取光譜數據，文件格式為兩列數據：波長和強度
    """
    # 使用 delimiter='\t' 來處理以 tab 分隔的數據
    data = np.loadtxt(file_path, delimiter='\t')
    wavelength = data[:, 0]
    intensity = data[:, 1]
    return wavelength, intensity


# 計算半高全寬 (FWHM)
def calculate_fwhm(wavelength, intensity, wavelength_range):
    """
    在指定的波長範圍內計算光譜的半高全寬 (FWHM)
    使用線性插值法來精確計算半高點的位置
    """
    # 限制在指定波長範圍內
    mask = (wavelength >= wavelength_range[0]) & (wavelength <= wavelength_range[1])
    wavelength_range = wavelength[mask]
    intensity_range = intensity[mask]

    # 找到光譜的最大值及其對應的波長
    max_intensity = np.max(intensity_range)
    max_index = np.argmax(intensity_range)
    max_wavelength = wavelength_range[max_index]

    # 計算半高
    half_max_intensity = max_intensity / 2

    # 使用插值法來找到半高位置
    interp_func = interp1d(wavelength_range, intensity_range, kind='linear')

    # 找到左半部的半高點
    left_mask = wavelength_range < max_wavelength
    left_wavelengths = wavelength_range[left_mask]
    left_intensities = intensity_range[left_mask]

    if np.any(left_intensities < half_max_intensity):
        left_interp = interp1d(left_intensities, left_wavelengths, kind='linear')
        left_fwhm = left_interp(half_max_intensity)
    else:
        left_fwhm = left_wavelengths[0]  # 無法找到半高點時，取最左端

    # 找到右半部的半高點
    right_mask = wavelength_range > max_wavelength
    right_wavelengths = wavelength_range[right_mask]
    right_intensities = intensity_range[right_mask]

    if np.any(right_intensities < half_max_intensity):
        right_interp = interp1d(right_intensities, right_wavelengths, kind='linear')
        right_fwhm = right_interp(half_max_intensity)
    else:
        right_fwhm = right_wavelengths[-1]  # 無法找到半高點時，取最右端

    # 計算 FWHM
    fwhm = right_fwhm - left_fwhm
    return fwhm, left_fwhm, right_fwhm, half_max_intensity


# 繪製光譜和半高全寬
def plot_spectrum(wavelength, intensity, wavelength_range, fwhm, left_fwhm, right_fwhm, half_max_intensity):
    plt.plot(wavelength, intensity, label='Spectrum')
    plt.axvline(left_fwhm, color='r', linestyle='--', label='FWHM Left')
    plt.axvline(right_fwhm, color='g', linestyle='--', label='FWHM Right')
    plt.axhline(half_max_intensity, color='b', linestyle='--', label='Half Max')
    plt.xlim(wavelength_range[0], wavelength_range[1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title(f'FWHM: {fwhm:.2f} nm')
    plt.legend()
    plt.show()


# 主程序
def main():
    # 載入光譜數據，替換成你光譜文件的路徑
    spectrum_file = 'Ocean.txt'
    wavelength, intensity = load_spectrum(spectrum_file)

    # 使用者輸入要分析的波長範圍
    start_wavelength = float(input("請輸入波長範圍起始值 (nm): "))
    end_wavelength = float(input("請輸入波長範圍結束值 (nm): "))
    wavelength_range = (start_wavelength, end_wavelength)

    # 計算半高全寬 (FWHM)
    fwhm, left_fwhm, right_fwhm, half_max_intensity = calculate_fwhm(wavelength, intensity, wavelength_range)

    # 顯示結果
    print(f"半高全寬 (FWHM): {fwhm:.2f} nm")
    print(f"左半高點: {left_fwhm:.2f} nm")
    print(f"右半高點: {right_fwhm:.2f} nm")

    # 繪製光譜和 FWHM
    plot_spectrum(wavelength, intensity, wavelength_range, fwhm, left_fwhm, right_fwhm, half_max_intensity)


if __name__ == "__main__":
    main()
