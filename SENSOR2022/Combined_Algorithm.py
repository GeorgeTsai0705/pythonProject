import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def load_data():
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    X_data = np.transpose(spectral_data.iloc[1:, 1:].values)
    Y_result = c_pass_result.iloc[3, 1:].values.astype(float)

    return X_data, Y_result

def create_threshold_labels(Y_result, threshold):
    return [1 if y > threshold else 0 for y in Y_result]

def apply_pca(X_data, n_components=5):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_data)

def train_classifier(X_train, y_train, model_class, model_params):
    best_model = None
    best_accuracy = 0

    for _ in range(5):
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_train, model.predict(X_train))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"Training {model_class.__name__}: Accuracy: {best_accuracy}")
    return best_model

def train_regressor(X_train, y_train, model_class, model_params):
    best_model = None
    best_mse = float('inf')
    best_r2 = -float('inf')

    for _ in range(5):
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        r2 = r2_score(y_train, predictions)

        # Update if the current model is better
        if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_model = model

    print(f"Training {model_class.__name__}: MSE: {best_mse}, R2 Score: {best_r2}")
    return best_model

def main():
    X_data, Y_result = load_data()
    X_data_pca = apply_pca(X_data)
    labels = create_threshold_labels(Y_result, threshold=np.median(Y_result))
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X_data_pca,
        Y_result,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    MLP_CLASSIFIER_PARAMS = {
        'hidden_layer_sizes': (7, 3),  # 隱藏層的大小。
        'activation': 'relu',  # 激活函數。可以是'identity', 'logistic', 'tanh', 或 'relu'。
        'solver': 'lbfgs',  # 優化權重的算法。可以是'lbfgs', 'sgd'或'adam'。
        'alpha': 0.1,  # L2懲罰的參數值（正則化項）。
        'batch_size': 'auto',  # 優化時的小批量的大小。
        'learning_rate': 'constant',  # 學習速率的計劃。
        'learning_rate_init': 0.001,  # 初始學習速率。
        'max_iter': 2500,  # 最大迭代次數。
        'shuffle': True,  # 在每次迭代時是否應隨機重排序樣本。
        'random_state': None,  # 狀態或種子的隨機生成。
        'early_stopping': True,
        'tol': 1e-4  # 優化的容忍度。
    }

    classifier = train_classifier(X_train, labels_train, MLPClassifier, MLP_CLASSIFIER_PARAMS)
    classifier_labels = classifier.predict(X_test)

    # 確保 X_train 是 DataFrame，且 labels_train 也是相同的索引
    X_train_df = pd.DataFrame(X_train)
    labels_train_df = pd.Series(labels_train, index=X_train_df.index)

    rf_X_train = X_train_df[labels_train_df == 1].values
    rf_y_train = y_train[labels_train_df == 1]

    mlp_X_train = X_train_df[labels_train_df == 0].values
    mlp_y_train = y_train[labels_train_df == 0]

    RF_REGRESSOR_PARAMS = {
        'n_estimators': 15,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': False
    }

    MLP_REGRESSOR_PARAMS = {
        'hidden_layer_sizes': (3,),
        'activation': 'relu',
        'solver': 'lbfgs',
        'alpha': 0.05,
        'learning_rate': 'adaptive',
        'max_iter': 3500,
        'early_stopping': True
    }

    rf_regressor = train_regressor(rf_X_train, rf_y_train, RandomForestRegressor, RF_REGRESSOR_PARAMS)
    mlp_regressor = train_regressor(mlp_X_train, mlp_y_train, MLPRegressor, MLP_REGRESSOR_PARAMS)

    combined_predictions = []
    for i, label in enumerate(classifier_labels):
        if label == 1:
            combined_predictions.append(rf_regressor.predict([X_test[i]])[0])
        else:
            combined_predictions.append(mlp_regressor.predict([X_test[i]])[0])

    # Visualization
    plt.figure(figsize=(12, 7))
    plt.scatter(y_test, combined_predictions, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
    plt.title("Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
