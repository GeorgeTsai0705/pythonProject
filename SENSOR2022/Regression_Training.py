import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
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
    Y_result = c_pass_result.iloc[3, 1:].values.ravel()

    return X_data, Y_result

def train_and_validate(model, params, X_train, y_train, X_test, y_test):
    results = []
    model_counter = 0
    for p in params:
        estimator = model(**p)
        start_time = time.time()
        estimator.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        y_pred = estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result = p.copy()
        result.update({
            'Training Time': training_time,
            'Test MSE': mse,
            'Test MAE': mae,
            'Test R2 Score': r2
        })
        results.append(result)

        model_counter += 1
        if model_counter % 5 == 0:
            print(f"已完成訓練 {model_counter} 個模型")

    return pd.DataFrame(results)

# Load Data
X_data, Y_result = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=42)

# PCA
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Parameter Grids
svr_param_grid = {
    'C': [5, 10, 20, 30, 40, 50, 60],
    'gamma': ['scale', 'auto', 0.7, 1, 2, 4],
    'kernel': ['linear', 'rbf', 'poly']
}

rf_param_grid = {
    'n_estimators': [2, 5, 10, 15, 20, 25],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

mlp_param_grid = {
    'hidden_layer_sizes': [(3,), (5,), (5, 3), (3, 2), (7, 3), (4, 2)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.01, 0.05, 0.1, 1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [2500],
    'early_stopping': [True]
}

# Train and Validate
svr_results = train_and_validate(SVR, ParameterGrid(svr_param_grid), X_train_pca, y_train, X_test_pca, y_test)
rf_results = train_and_validate(RandomForestRegressor, ParameterGrid(rf_param_grid), X_train_pca, y_train, X_test_pca, y_test)
mlp_results = train_and_validate(MLPRegressor, ParameterGrid(mlp_param_grid), X_train_pca, y_train, X_test_pca, y_test)

# Save results
svr_results.sort_values(by=['Test R2 Score', 'Test MSE', 'Training Time'], ascending=[False, True, True]).to_csv('Grid_Result\Combined_SVR_evaluation_sorted_results.csv', index=False)
rf_results.sort_values(by=['Test R2 Score', 'Test MSE', 'Training Time'], ascending=[False, True, True]).to_csv('Grid_Result\Combined_RF_evaluation_sorted_results.csv', index=False)
mlp_results.sort_values(by=['Test R2 Score', 'Test MSE', 'Training Time'], ascending=[False, True, True]).to_csv('Grid_Result\Combined_MLP_evaluation_sorted_results.csv', index=False)
