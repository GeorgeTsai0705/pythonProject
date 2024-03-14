import cv2
import numpy as np
import time

# 打開預設的攝像頭
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_MODE, 2)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# 檢查攝像頭是否成功打開
if not cap.isOpened():
    print("無法打開攝像頭")
    exit()

while True:
    # 讀取一幀影像
    ret, frame = cap.read()

    # 檢查是否成功讀取到影像
    if not ret:
        print("無法讀取影像")
        break

    start_time = time.time()  # 開始計時
    """
    combined = []
    for i in range(0, len(frame[0]), 2):
        a = frame[0][i]
        b = frame[0][i + 1] if i + 1 < len(frame[0]) else 0  # 確保不會超出範圍
         # 結合a的全部8位和b的後4位
        ele = (a << 4) | (b - 128)
        combined.append(ele)
    result = np.array(combined).reshape(800, 1280)
    normalized_result = result / 4096 * 255
    normalized_result = normalized_result.astype(np.uint8)

    end_time = time.time()  # 結束計時
    """

    a = frame[0][::2].astype(np.uint16)  # 從0開始，每2個元素取一個
    b = frame[0][1::2].astype(np.uint16)  # 從1開始，每2個元素取一個
    if len(frame[0]) % 2 != 0:
        b = np.append(b, 0)  # 確保a和b的長度相同

    combined2 = ((a << 4) | (b - 128)).astype(np.uint16)
    result2 = combined2.reshape(800, 1280)
    new_result2 = result2.astype(np.float64)
    normalized_result2 = new_result2 / 4096 * 254
    normalized_result2 = normalized_result2.astype(np.uint8)
    
    end_time = time.time()  # 結束計時

    print("處理這幀影像所需的時間：{:.2f}秒".format(end_time - start_time))

    # 使用NumPy數組顯示讀取結果
    #print("影像尺寸:", frame.shape)
    #print("影像數據類型:", frame.dtype)

    resized_image = cv2.resize(normalized_result2, (0,0),fx=0.5, fy=0.5)
    cv2.imshow('result_image', resized_image)

    # 等待1毫秒，檢查是否有鍵被按下，按'q'退出
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# 釋放攝像頭資源並關閉所有窗口
cap.release()
cv2.destroyAllWindows()
