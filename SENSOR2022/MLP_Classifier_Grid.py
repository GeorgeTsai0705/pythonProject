import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neural_network import MLPClassifier
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
indices = np.arange(len(X_data))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X_data, Y_result, indices, test_size=0.2, random_state=42)

# 定義要進行grid搜尋的參數
param_grid = {
    'hidden_layer_sizes': [(3,), (5,), (5, 3), (3, 2), (7, 3), (4, 2)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [ 0.01, 0.05, 0.1, 1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [2500],
    'early_stopping': [True]
}

results = []
model_counter = 0
for params in ParameterGrid(param_grid):
    mlp = MLPClassifier(**params)
    start_time = time.time()
    mlp.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = mlp.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).ravel()  # 將混淆矩陣轉為1D
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    results.append({
        'hidden_layer_sizes': params['hidden_layer_sizes'],
        'activation': params['activation'],
        'solver': params['solver'],
        'alpha': params['alpha'],
        'learning_rate': params['learning_rate'],
        'Training Time': training_time,
        'Test Accuracy': acc,
        'Test Precision': prec,
        'Test Recall': rec,
        'Test F1 Score': f1,
        'Confusion Matrix': list(cm),
        'Test Indices': test_indices.tolist()  # 把 numpy array 轉成 list
    })

    model_counter += 1
    if model_counter % 5 == 0:
        print(f"已完成訓練 {model_counter} 個模型")

results_df = pd.DataFrame(results)

# 根據 Accuracy、F1 Score、Training Time 進行排序
results_df = results_df.sort_values(by=['Test Accuracy', 'Test F1 Score', 'Training Time'],
                                    ascending=[False, False, True])

results_df.to_csv('Grid_Result/mlp_evaluation_sorted_results.csv', index=False)
