import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import joblib
import json


def find_best_thresholds_for_3classes(scores, true_labels):
    """
    給定每筆資料的分數 scores 與真實標籤 true_labels (A/B/C)，
    暴力搜尋兩個 threshold t1 < t2，
    分別把分數劃分成 [C 區間], [B 區間], [A 區間]，
    回傳能達到最高 accuracy 的 (best_t1, best_t2, best_acc)。
    """

    # 1. 先把分數做唯一化與排序
    unique_scores = np.unique(scores)

    # 若資料分數全都一樣或太少，就直接退出
    if len(unique_scores) < 3:
        return None, None, 0.0

    best_acc = 0.0
    best_t1, best_t2 = None, None

    # 2. 兩層迴圈暴力搜尋
    #    為了避免直接在分數上切，常會用 (score_i + score_(i+1)) / 2
    #    做中點以避免區分不開
    for i in range(len(unique_scores) - 2):
        t1 = (unique_scores[i] + unique_scores[i + 1]) / 2.0

        for j in range(i + 1, len(unique_scores) - 1):
            t2 = (unique_scores[j] + unique_scores[j + 1]) / 2.0

            # 確保 t1 < t2
            if t1 >= t2:
                continue

            # 3. 根據 (t1, t2) 進行預測
            pred_labels = []
            for s in scores:
                if s < t1:
                    pred_labels.append("C")
                elif s < t2:
                    pred_labels.append("B")
                else:
                    pred_labels.append("A")

            # 4. 計算預測與真實標籤的正確率
            acc = accuracy_score(true_labels, pred_labels)

            # 5. 若比目前最佳還好，就更新
            if acc > best_acc:
                best_acc = acc
                best_t1, best_t2 = t1, t2

    return best_t1, best_t2, best_acc


# 1. 載入 CSV 檔案
file_path = 'D:/光核心/2025.01.02 宜運光譜儀/chip_summary.csv'
data = pd.read_csv(file_path)

# 2. 選取相關欄位（假設跟原程式相同）
columns_of_interest = [
    'fwhm_mean', 'feature1_ratio', 'feature2_ratio',
    'feature3_ratio', 'feature4_difference','feature5', 'feature6','fwhm_std', 'Baseline Evaluation'
]

# 3. 從資料中擷取子集
data_subset = data[columns_of_interest]

# 4. 記錄正規化參數（最大值和最小值），以便未來需要時可回復或應用在新資料
normalization_params = {
    column: {
        'max': data_subset[column].max(),
        'min': data_subset[column].min()
    }
    for column in columns_of_interest
}

# 5. 將每個欄位正規化至 [0, 1] 範圍，且「數值越小越好」的邏輯保持不變
normalized_data = (data_subset.max() - data_subset) / (data_subset.max() - data_subset.min())

# ===== (此區僅示範印出正規化參數，非必要可自行註解) =====
print("Normalization Parameters:")
for column, params in normalization_params.items():
    print(f"{column}: max = {params['max']}, min = {params['min']}")
# =====================================================

# 6. LabelEncoder 將多類別 (A / B / C) 轉為整數 (0 / 1 / 2)
label_encoder = LabelEncoder()
data['encoded_Class'] = label_encoder.fit_transform(data['Class'])
# 例如：若 label_encoder.classes_ = ['A', 'B', 'C']，那麼
#  'A' -> 0, 'B' -> 1, 'C' -> 2 （實際對應順序視資料而定）

# 7. 準備特徵 X 以及多類別標籤 y
X = normalized_data
y = data['encoded_Class']

# 8. 訓練 / 測試分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. 建立並訓練隨機森林
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 10. 提取特徵重要性並轉換為「權重百分比」
feature_importances = rf.feature_importances_
importance_sum = np.sum(feature_importances)
weights = {
    columns_of_interest[i]: feature_importances[i] / importance_sum
    for i in range(len(columns_of_interest))
}

# 11. 輸出每個特徵的權重
print("Optimized Feature Weights:")
for feature, weight in weights.items():
    print(f"{feature}: {weight:.4f}")

# 12. 計算每一筆資料的「加權分數」(僅作範例：即把所有特徵的 normalized 值乘以其權重後加總)
#    如果要一次對多類別做評估，可視需求進行調整：例如分別計算對各類別的 score 等。
normalized_data['score'] = normalized_data.apply(
    lambda row: sum(row[col] * weights[col] for col in columns_of_interest),
    axis=1
)

# 將分數添加回原始資料 (非必須，但若想要對照可以保留)
data['score'] = normalized_data['score']

scores = data['score'].values
true_labels = data['Class'].values

t1, t2, best_acc = find_best_thresholds_for_3classes(scores, true_labels)

print("Best T1 =", t1)
print("Best T2 =", t2)
print("Best Accuracy =", best_acc)

# 取得最終預測
pred_labels = []
for s in scores:
    if s < t1:
        pred_labels.append("C")
    elif s < t2:
        pred_labels.append("B")
    else:
        pred_labels.append("A")

# 加回原 DataFrame
data['predicted_class_by_score'] = pred_labels

# 13. 使用隨機森林做預測
y_pred = rf.predict(X_test)

# 14. 將數字標籤轉回原始 A/B/C 標籤，以便做報表或混淆矩陣可讀性
y_test_classes = label_encoder.inverse_transform(y_test)
y_pred_classes = label_encoder.inverse_transform(y_pred)

# 15. 計算多類別 ROC-AUC
#    - 若資料量足夠並想做 one-vs-one，可使用 multi_class='ovo'
#    - 若想做 one-vs-rest，可使用 multi_class='ovr'
y_score = rf.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')  # 或 'ovo'
joblib.dump(rf, "random_forest_model.pkl")

# 16. 多類別分類報告與混淆矩陣
print("ROC AUC (multi-class, OVR):", roc_auc)
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))

# 17. 將最終結果（含預測類別）存回原始 data 裡，為了對齊要把整份資料都做 predict
#    - 這邊示範對整份 data 做預測 (train+test)，視需求也可只存測試集
full_pred = rf.predict(normalized_data[columns_of_interest])
print(full_pred)
data['predicted_Class'] = label_encoder.inverse_transform(full_pred)

# 18. 將分數與分類結果保存到 CSV
output_file_path = 'D:/光核心/2025.01.02 宜運光譜儀/ALL_with_scores_and_predictions.csv'
data.to_csv(output_file_path, index=False)
print(f"Data with scores and predictions saved to {output_file_path}")

results = {
    feature: {
        "max": normalization_params[feature]["max"],
        "min": normalization_params[feature]["min"],
        # Round the weight as needed (here to 4 decimal places)
        "weight": round(weights[feature], 4)
    }
    for feature in columns_of_interest
}

print(json.dumps(results, indent=4))