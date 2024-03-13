import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # 標準化資料

def load_data():
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    X_data = np.transpose(spectral_data.iloc[1:, 1:].values)
    Y_result = c_pass_result.iloc[3, 1:].values.astype(float)

    return X_data, Y_result

def create_threshold_labels(Y_result, threshold_1, threshold_2):
    return [1 if threshold_2 > y > threshold_1 else 0 for y in Y_result]



def apply_pca(X_data, n_components=5):
    # 先進行標準化
    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data)

    pca = PCA(n_components=0.99)
    return pca.fit_transform(X_data_scaled)


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

    # 計算和打印驗證指標
    y_pred = best_model.predict(X_train)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    conf_matrix = confusion_matrix(y_train, y_pred)

    print(f"Training {model_class.__name__}:")
    print(f"Accuracy: {best_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)

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

    print(f"Training {model_class.__name__}: MSE: {best_mse}, R2 Score: {best_r2:.3f}")
    return best_model


def main():

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
    X_data, Y_result = load_data()
    X_data_pca = apply_pca(X_data)
    print(f"PCA Result {np.shape(X_data_pca)}")

    best_mae = float('inf')
    best_threshold = None
    best_combined_predictions = None
    best_ytest = None
    best_R2score = None

    for threshold_1 in range(30, 80, 3):
        for threshold_2 in range(130, 181, 3):
            if threshold_1 != threshold_2:  # 只有当两个阈值不相等时才进行计算
                labels = create_threshold_labels(Y_result, threshold_1, threshold_2)

                X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
                    X_data_pca,
                    Y_result,
                    labels,
                    test_size=0.2,
                    random_state=47,
                    stratify=labels
                )

                classifier = train_classifier(X_train, labels_train, MLPClassifier, MLP_CLASSIFIER_PARAMS)
                classifier_labels = classifier.predict(X_test)

                X_train_df = pd.DataFrame(X_train)
                labels_train_df = pd.Series(labels_train, index=X_train_df.index)

                rf_X_train = X_train_df[labels_train_df == 1].values
                rf_y_train = y_train[labels_train_df == 1]
                mlp_X_train = X_train_df[labels_train_df == 0].values
                mlp_y_train = y_train[labels_train_df == 0]

                rf_regressor = train_regressor(rf_X_train, rf_y_train, RandomForestRegressor, RF_REGRESSOR_PARAMS)
                mlp_regressor = train_regressor(mlp_X_train, mlp_y_train, MLPRegressor, MLP_REGRESSOR_PARAMS)

                combined_predictions = []
                for i, label in enumerate(classifier_labels):
                    if label == 1:
                        combined_predictions.append(rf_regressor.predict([X_test[i]])[0])
                    else:
                        combined_predictions.append(mlp_regressor.predict([X_test[i]])[0])

                mae = mean_absolute_error(y_test, combined_predictions)
                print(f"Threshold: {threshold_1, threshold_2}, Combined Model MAE: {mae}")

                if mae < best_mae:
                    best_mae = mae
                    best_threshold = [threshold_1, threshold_2]
                    best_combined_predictions = combined_predictions
                    best_ytest = y_test
                    best_R2score = r2_score(y_test, best_combined_predictions)

    print(f"Best Threshold: {best_threshold}, Best MAE: {best_mae}, Best R2: {best_R2score}")

    # 若需要，您還可以視覺化最佳模型的結果
    plt.figure(figsize=(12, 7))
    plt.scatter(best_ytest, best_combined_predictions, alpha=0.5)
    plt.plot([min(best_ytest), max(best_ytest)], [min(best_ytest), max(best_ytest)], color='red')  # Diagonal line
    plt.title("Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()

    # 將最佳模型的結果存儲到DataFrame
    results_df = pd.DataFrame({
        'Actual_Values': best_ytest,
        'Predicted_Values': best_combined_predictions
    })

    # 將DataFrame輸出到CSV文件
    results_df.to_csv('Grid_Result/best_model_results.csv', index=False)

    print("Best model results saved to 'best_model_results.csv'.")


if __name__ == "__main__":
    main()
