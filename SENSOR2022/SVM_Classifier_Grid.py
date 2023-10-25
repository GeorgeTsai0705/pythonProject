import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

results = []
for params in ParameterGrid(param_grid):
    svm = SVC(**params)
    start_time = time.time()
    svm.fit(X_train_pca, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = svm.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    results.append({
        'C': params['C'],
        'gamma': params['gamma'],
        'kernel': params['kernel'],
        'Training Time': training_time,
        'Test Accuracy': acc,
        'Test Precision': prec,
        'Test Recall': rec,
        'Test F1 Score': f1
    })

results_df = pd.DataFrame(results)
results_df.to_csv('svm_evaluation_results.csv', index=False)
