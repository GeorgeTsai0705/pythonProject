import cv2
import numpy as np

# 打開預設的攝像頭
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_MODE, 2)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
# 檢查攝像頭是否成功打開
if not cap.isOpened():
    print("無法打開攝像頭")
    exit()

# 讀取一幀影像
ret, frame = cap.read()

combined = []

for i in range(0, len(frame[0]), 2):
    a = frame[0][i]
    b = frame[0][i+1] if i + 1 < len(frame[0]) else 0  # 確保不會超出範圍
    # 結合a的全部8位和b的後4位
    ele = (a << 4) | (b-128)
    combined.append(ele)
result = np.array(combined).reshape(800, 1280)
normalized_result = result / 4096 * 255
normalized_result = normalized_result.astype(np.uint8)

# 檢查是否成功讀取到影像
if not ret:
    print("無法讀取影像")
    exit()
# 使用NumPy數組顯示讀取結果
print("影像尺寸:", frame.shape)
print("影像數據類型:", frame.dtype)

cv2.imwrite('result_image.png', normalized_result)
cv2.imshow('result_image', normalized_result)
# 等待按鍵事件，然後關閉窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 釋放攝像頭資源
cap.release()
