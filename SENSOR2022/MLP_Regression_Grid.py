import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def load_data():
    """加載數據"""
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    wave_length = spectral_data.iloc[1:, 0].tolist()
    X_data = np.transpose(spectral_data.iloc[1:, 1:])

    sample_names = c_pass_result.iloc[0, 1:]
    Y_result = c_pass_result.iloc[3, 1:].values.ravel()  # 將 Y_result 轉為 1D 數組

    return X_data, Y_result

# === 主程式區域 ===
X_data, Y_result = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=42)

# 定義要進行grid搜尋的參數
param_grid = {
    'hidden_layer_sizes': [(3,), (5,), (5, 3), (3, 2), (7, 3), (4, 2)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.01, 0.05, 0.1, 1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [2500],
    'early_stopping': [True]
}

results = []
model_counter = 0
for params in ParameterGrid(param_grid):
    mlp = MLPRegressor(**params)
    start_time = time.time()
    mlp.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = mlp.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'hidden_layer_sizes': params['hidden_layer_sizes'],
        'activation': params['activation'],
        'solver': params['solver'],
        'alpha': params['alpha'],
        'learning_rate': params['learning_rate'],
        'Training Time': training_time,
        'Test MSE': mse,
        'Test MAE': mae,
        'Test R2 Score': r2
    })

    model_counter += 1
    if model_counter % 5 == 0:
        print(f"已完成訓練 {model_counter} 個模型")

results_df = pd.DataFrame(results)

# 根據 R2 Score、MSE、Training Time 進行排序
results_df = results_df.sort_values(by=['Test R2 Score', 'Test MSE', 'Training Time'],
                                    ascending=[False, True, True])

results_df.to_csv('Grid_Result/mlp_regression_evaluation_sorted_results.csv', index=False)
