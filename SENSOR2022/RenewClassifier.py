import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# === 函式區域 ===

def load_data():
    """加載數據"""
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    wave_length = spectral_data.iloc[1:, 0].tolist()
    X_data = np.transpose(spectral_data.iloc[1:, 1:])

    sample_names = c_pass_result.iloc[0, 1:]
    Y_result = c_pass_result.iloc[2, 1:].values.ravel()  # 將 Y_result 轉為 1D 數組

    return X_data, Y_result


def plot_pca_variance(explained_variance):
    """繪製PCA的解釋方差比"""
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(len(cumulative_explained_variance)), cumulative_explained_variance, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """訓練和評估多個模型"""
    metrics_data = {
        'Model': [],
        'Training Time': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 5))

    for idx, (model_name, model) in enumerate(models.items()):
        # 訓練模型
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        # 評估模型
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Store metrics
        metrics_data['Model'].append(model_name)
        metrics_data['Training Time'].append(end_time - start_time)
        metrics_data['Accuracy'].append(acc)
        metrics_data['Precision'].append(prec)
        metrics_data['Recall'].append(rec)
        metrics_data['F1 Score'].append(f1)

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm)
        sns.heatmap(cm_df, annot=True, fmt='g', ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix for {model_name}')
        axes[idx].set_ylabel('True label')
        axes[idx].set_xlabel('Predicted label')

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(metrics_data)


# === 主程式區域 ===
X_data, Y_result = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=42)

# 套用 PCA 降維
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

plot_pca_variance(pca.explained_variance_ratio_)


# SVM 參數
svm_params = {
    'C': 100,  # 正則化參數。值越小，模型的正則化強度越大。
    'kernel': 'linear',  # 指定算法的核。選項有'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'或可調用。
    'degree': 3,  # 如果kernel設置為'poly'，則它是多項式的度數。
    'gamma': 'auto',  # 'poly', 'rbf', 'sigmoid'的核的係數。'scale'或'auto'。
    'coef0': 0.0,  # kernel函數中的獨立項。只對'poly'和'sigmoid'有效。
    'shrinking': True  # 是否使用shrinking heuristic（縮小啟發式方法）。
}

# 隨機森林參數
rf_params = {
    'n_estimators': 5,  # 森林中樹的數量。
    'criterion': 'gini',  # 分裂質量的功能。可以是"gini"或"entropy"。
    'max_depth': 3,  # 樹的最大深度。
    'min_samples_split': 5,  # 分裂內部節點所需的最小樣本數。
    'min_samples_leaf': 1,  # 葉節點所需的最小樣本數。
    'max_features': 'sqrt',  # 尋找最佳分裂時要考慮的特徵數。
    'bootstrap': True,  # 是否在構建樹時使用bootstrap樣本。
    'oob_score': False,  # 是否使用out-of-bag樣本來估計泛化精度。
    'n_jobs': -1,  # 適合和預測的作業數。
    'random_state': None  # 隨機數生成的種子。
}

# MLP 參數
mlp_params = {
    'hidden_layer_sizes': (7,3),  # 隱藏層的大小。
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


models = {
    'SVM': svm.SVC(**svm_params),
    'RandomForest': RandomForestClassifier(**rf_params),
    'MLP': MLPClassifier(**mlp_params)
}

metrics_df = train_and_evaluate(models, X_train_pca, y_train, X_test_pca, y_test)
print(metrics_df)
