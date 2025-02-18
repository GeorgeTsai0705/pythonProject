import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import numpy as np

# 載入 CSV 檔案
file_path = 'D:/光核心/2025.01.02 宜運光譜儀/ALL.csv'
data = pd.read_csv(file_path)

# 選取相關欄位
columns_of_interest = [
    'fwhm_mean', 'feature1_ratio', 'feature2_ratio',
    'feature3_ratio', 'feature4_difference','feature5', 'fwhm_std', 'Baseline Evaluation'
]

# 從資料中擷取子集
data_subset = data[columns_of_interest]

# 記錄正規化參數（最大值和最小值）
normalization_params = {
    column: {
        'max': data_subset[column].max(),
        'min': data_subset[column].min()
    }
    for column in columns_of_interest
}

# 將每個欄位正規化至 [0, 1] 範圍，數值越小越好
normalized_data = (data_subset.max() - data_subset) / (data_subset.max() - data_subset.min())

# 列印正規化參數
print("Normalization Parameters:")
for column, params in normalization_params.items():
    print(f"{column}: max = {params['max']}, min = {params['min']}")

# 訓練隨機森林模型以獲取特徵重要性
true_labels = (data['Class'] == 'A').astype(int)  # 將類別 A 編碼為 1，類別 B 編碼為 0
X_train, X_test, y_train, y_test = train_test_split(normalized_data, true_labels, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 提取特徵重要性並正規化為權重
feature_importances = rf.feature_importances_
weights = {columns_of_interest[i]: feature_importances[i] / sum(feature_importances) for i in range(len(columns_of_interest))}

# 輸出每個特徵的權重
print("Optimized Feature Weights:")
for feature, weight in weights.items():
    print(f"{feature}: {weight:.4f}")

# 計算每一筆資料的分數
normalized_data['score'] = normalized_data.apply(
    lambda row: sum(row[col] * weights[col] for col in columns_of_interest), axis=1
)

# 將分數添加回原始資料
data['score'] = normalized_data['score']

# 計算 ROC 和 AUC
roc_auc = roc_auc_score(true_labels, data['score'])

# 搜尋最佳 threshold
fpr, tpr, thresholds = roc_curve(true_labels, data['score'])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# 透過最佳 threshold 評估 AB 類別
data['predicted_class'] = data['score'].apply(lambda x: 'A' if x >= optimal_threshold else 'B')

# 計算與真實類別的混淆矩陣和報告
classification_report_result = classification_report(data['Class'], data['predicted_class'])
confusion_matrix_result = confusion_matrix(data['Class'], data['predicted_class'])

print("Optimal Threshold:", optimal_threshold)
print("ROC AUC:", roc_auc)
print("Classification Report:\n", classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)

# 將分數與分類結果保存到 CSV
output_file_path = 'D:/光核心/2025.01.02 宜運光譜儀/ALL_with_scores_and_predictions.csv'
data.to_csv(output_file_path, index=False)
print(f"Data with scores and predictions saved to {output_file_path}")
