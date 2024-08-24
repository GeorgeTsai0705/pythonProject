import cv2
import tkinter as tk
from tkinter import filedialog
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

if __name__ == "__main__":
    # 初始化Tkinter並選擇文件
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    sVideo = filedialog.askopenfilename(title="Select Video File",
                                        filetypes=(("Video Files", "*.mp4;*.avi;*.h264"), ("All Files", "*.*")))

    if sVideo:  # 確保使用者選擇了檔案
        cap = cv2.VideoCapture(sVideo)

        # 生成儲存數據的txt文件名
        base_filename = os.path.splitext(os.path.basename(sVideo))[0]
        output_filename = base_filename + "_data.txt"

        # 打开文件，准备写入数据
        with open(output_filename, 'w') as f:

            # 計算影片的總幀數
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.release()

            # 重新打開影片文件
            cap = cv2.VideoCapture(sVideo)
            delay = 10 / 1000.0  # 20毫秒的延遲
            current_frame = 0  # 初始化當前幀數

            # 初始化圖表
            plt.ion()
            fig, ax = plt.subplots()

            # 設定要處理的行區域
            row_start = 250  # 設定行區域開始位置
            row_end = 265    # 設定行區域結束位置

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame += 1  # 增加當前幀數

                # 將幀轉換為灰階影像
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ROI影像處理，取特定行區域
                roi = gray_frame[row_start:row_end, :]  # 提取指定行區域
                col_avg = np.mean(roi, axis=0)  # 對每一列取平均值

                # 使用SG濾波器平滑化col_avg
                window_length = 15  # 窗口長度，必須為奇數
                polyorder = 3       # 多項式階數
                col_avg_smoothed = savgol_filter(col_avg, window_length, polyorder)

                # 儲存col_avg_smoothed到txt文件
                np.savetxt(f, [col_avg_smoothed], fmt='%.6f', delimiter=',')

                # 在影像上畫出紅色的ROI框
                cv2.rectangle(frame, (0, row_start), (gray_frame.shape[1], row_end), (0, 0, 255), 1)

                # 更新圖表數據
                ax.clear()
                ax.plot(col_avg_smoothed, label=f'Frame {current_frame}')
                ax.set_xlabel('Column Index')
                ax.set_ylabel('Column Average Intensity')
                ax.set_title('Column Average Intensity Across ROI (Gray)')
                ax.set_ylim(0,200)
                ax.legend()
                plt.pause(0.01)  # 更新圖表

                # 在frame上標示當前幀數
                text = f"Frame: {current_frame}/{total_frames}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # 新增顯示ROI位置的文字
                text_roi = f"ROI: Rows {row_start}-{row_end}"
                cv2.putText(frame, text_roi, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # 縮小影像顯示
                resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

                cv2.imshow('Video Playback', resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下q鍵退出
                    break

                time.sleep(delay)  # 等待20毫秒

            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.show()
