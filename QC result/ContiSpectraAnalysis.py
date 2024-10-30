import numpy as np
import os
import re
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog
from datetime import datetime


# 讀取光譜數據
def load_spectrum(file_path):
    """
    從文件中讀取光強數據，只有一列代表各個像素的光強。
    """
    intensity = np.loadtxt(file_path)
    return intensity


# 從檔名中提取時間戳和序列號
def extract_timestamp_and_sequence(filename):
    """
    從檔名中提取時間戳和序列號，檔名格式例如 spectrum_0001_20240831_202404.txt。
    去除 .txt 副檔名
    """
    filename = filename.replace('.txt', '')  # 移除 .txt 副檔名
    match = re.match(r'spectrum_(\d+)_\d{8}_\d{6}', filename)
    if match:
        sequence = int(match.group(1))
        timestamp = filename.split('_')[-2] + "_" + filename.split('_')[-1]
        return sequence, timestamp
    return None, None


# 將時間戳記轉換為 datetime 物件
def parse_timestamp(timestamp_str):
    """
    將時間戳記字串（yyyymmdd_hhmmss）轉換為 datetime 物件。
    """
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")


# 計算從第一個時間點開始經過的秒數
def calculate_elapsed_seconds(timestamps):
    """
    根據時間戳記計算經過的秒數。
    """
    first_time = parse_timestamp(timestamps[0])
    elapsed_seconds = [(parse_timestamp(ts) - first_time).total_seconds() for ts in timestamps]
    return elapsed_seconds


# 主程序
def main():
    # 使用 Tkinter 讓使用者選擇資料夾
    root = Tk()
    root.withdraw()  # 隱藏主視窗
    folder_path = filedialog.askdirectory(title="請選擇光譜資料夾")

    if not folder_path:
        print("未選擇資料夾，程式結束。")
        return

    # 取得資料夾中所有的光譜檔案
    files = sorted([f for f in os.listdir(folder_path) if f.startswith('spectrum_') and f.endswith('.txt')])

    if len(files) == 0:
        print("未找到任何光譜檔案，程式結束。")
        return

    intensities = []
    timestamps = []

    # 載入所有光譜檔案
    for file in files:
        file_path = os.path.join(folder_path, file)

        # 載入光譜數據
        intensity = load_spectrum(file_path)
        intensities.append(intensity)

        # 提取時間戳
        _, timestamp = extract_timestamp_and_sequence(file)
        timestamps.append(timestamp)

    # 確認每個檔案的像素數相同
    pixel_count = len(intensities[0])
    print(f"每個光譜檔案有 {pixel_count} 個像素點。")

    # 使用者輸入要分析的行數（第幾個 row）
    target_row = int(input(f"請輸入要分析的行數 (0 ~ {pixel_count - 1}): "))

    # 確保範圍合法，取前後各 2 個 row，總共 5 個 row
    if target_row < 2 or target_row > pixel_count - 3:
        print(f"行數超出範圍，請確保行數在 2 到 {pixel_count - 3} 之間。")
        return

    # 紀錄最強光強的 row 位置和強度
    strongest_rows = []
    strongest_intensities = []

    # 進行連續分析
    for intensity in intensities:
        # 取出包含 target_row 前後兩個 row 的光強
        surrounding_rows = intensity[target_row - 2:target_row + 3]

        # 找到最強的 row 及其光強
        max_intensity = np.max(surrounding_rows)
        max_row_offset = np.argmax(surrounding_rows)  # 得到相對於 target_row 的偏移量
        max_row = target_row - 2 + max_row_offset

        # 紀錄最強的 row 位置和光強
        strongest_rows.append(max_row)
        strongest_intensities.append(max_intensity)

    # 計算每個時間點相對於第一個時間點的經過秒數
    elapsed_seconds = calculate_elapsed_seconds(timestamps)

    # 輸出結果到 .txt 檔案
    output_file = os.path.join(folder_path, "analysis_results.txt")
    with open(output_file, 'w') as f:
        f.write("Elapsed_Time(s)\tRow\tIntensity\n")
        for elapsed, row, intensity in zip(elapsed_seconds, strongest_rows, strongest_intensities):
            f.write(f"{elapsed:.2f}\t{row}\t{intensity:.2f}\n")

    print(f"結果已保存到 {output_file}")

    # 繪製最強 row 的變化情況
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(elapsed_seconds, strongest_rows, marker='o')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Row with Max Intensity')
    plt.title('Row with Maximum Intensity Over Time')

    # 繪製最強光強的變化情況
    plt.subplot(2, 1, 2)
    plt.plot(elapsed_seconds, strongest_intensities, marker='o')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Maximum Intensity')
    plt.title('Maximum Intensity Over Time')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
