import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data():
    """加載數據"""
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    wave_length = spectral_data.iloc[1:, 0].astype(float).tolist() # 確保波長是浮點數
    X_data = np.transpose(spectral_data.iloc[1:, 1:].astype(float).values) # 確保數據是浮點數

    Y_result = c_pass_result.iloc[3, 1:].astype(float).values.ravel()  # 將 Y_result 轉為 1D 數組

    return X_data, Y_result, wave_length

# 加載數據
X_data, Y_result, wave_length = load_data()

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化 PCA
pca = PCA(n_components=0.99)

# 擬合 PCA
pca.fit(X_train_scaled)

# 轉換數據
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_var_ratios = pca.explained_variance_ratio_

# 打印每个主成分的解释的方差比例
for i, ratio in enumerate(explained_var_ratios, start=1):
    print(f'Component {i}: {ratio:.4f} ({ratio:.2%})')

# 繪製貢獻度最大的幾個主成分
plt.figure(figsize=(12, 6))
sns.barplot(x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))], y=pca.explained_variance_ratio_)
plt.title('Explained Variance Ratio of PCA Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# 將主成分與原始特徵相關聯
component_matrix = pd.DataFrame(pca.components_, columns=wave_length)

num_components = 3  # 選擇展示的主成分數量
fig, axes = plt.subplots(num_components, 1, figsize=(12, num_components * 4))

# 計算每5個波長中的索引
indices = np.arange(0, len(wave_length), len(wave_length) // 5)

for i, (ax, comp) in enumerate(zip(axes, component_matrix.values[:num_components]), start=1):
    sns.barplot(x=wave_length, y=comp, ax=ax)

    # 只在選定的索引處設置標籤
    ax.set_xticks(indices)
    ax.set_xticklabels([wave_length[j] for j in indices], rotation=45, ha='right')  # 添加rotation以改善標籤的可讀性

    ax.set_title(f'Component {i}')
    ax.set_ylabel('Weight')
    ax.set_xlabel('Wavelength')

plt.tight_layout()
plt.show()

#選擇需要輸出到CSV的主成分數量
num_components_to_output = len(component_matrix.values)

# 為每個主成分創建DataFrame並輸出到CSV
for i, comp in enumerate(component_matrix.values[:num_components_to_output], start=1):
    # 創建一個包含波長和對應權重的DataFrame
    df = pd.DataFrame({'Wavelength': wave_length, f'Component_{i}_Weight': comp})

    # 輸出DataFrame到CSV
    df.to_csv(f'Component_{i}_Weights.csv', index=False)

    print(f'Component {i} weights saved to Component_{i}_Weights.csv')


# 繪製前兩個主成分
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('First Main Component')
plt.ylabel('Second Main Component')
plt.colorbar(label='Y value')
plt.title('PCA - First Two Components')
plt.show()
