import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time


def load_data():
    """加載數據"""
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    wave_length = spectral_data.iloc[1:, 0].tolist()
    X_data = np.transpose(spectral_data.iloc[1:, 1:])

    sample_names = c_pass_result.iloc[0, 1:]
    Y_result = c_pass_result.iloc[2, 1:].values.ravel()  # 將 Y_result 轉為 1D 數組

    return X_data, Y_result


# === 主程式區域 ===
X_data, Y_result = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=42)

# 套用 PCA 降維
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 定義要進行grid搜尋的參數
param_grid = {
    'n_estimators': [2, 5, 10, 15, 20, 25],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

results = []
model_counter = 0
for params in ParameterGrid(param_grid):
    rf = RandomForestClassifier(**params)
    start_time = time.time()
    rf.fit(X_train_pca, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = rf.predict(X_test_pca)

    cm = confusion_matrix(y_test, y_pred).ravel()  # 將混淆矩陣轉為1D
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    results.append({
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'bootstrap': params['bootstrap'],
        'Training Time': training_time,
        'Test Accuracy': acc,
        'Test Precision': prec,
        'Test Recall': rec,
        'Test F1 Score': f1,
        'Confusion Matrix': list(cm)
    })

    model_counter += 1
    if model_counter % 5 == 0:
        print(f"已完成訓練 {model_counter} 個模型")

results_df = pd.DataFrame(results)

# 根據 Accuracy、F1 Score、Training Time 進行排序
results_df = results_df.sort_values(by=['Test Accuracy', 'Test F1 Score', 'Training Time'],
                                    ascending=[False, False, True])

results_df.to_csv('Grid_Result/rf_evaluation_sorted_results.csv', index=False)
