#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt

# 讀取訓練和測試數據集
df = pd.read_csv('Citation_trianing_datav2.csv', encoding='unicode_escape')
dt = pd.read_csv('Citation_testing_data_v2.csv', encoding='unicode_escape')

# 從數據集中獲取x和y的值
X_train, Y_train = df[["2017","2018","2019", "2020", "2021"]].to_numpy(), df["Score"].to_numpy()
X_test, Y_test = dt[["2017","2018","2019", "2020", "2021"]].to_numpy(), dt["Score"].to_numpy()

def predict_value(model, x_train, y_train, x_test, y_test):
    # 訓練模型並預測輸入對應的y值
    model.fit(x_train, y_train)
    print(f"{type(model).__name__}: ")
    print(model.predict(x_test))
    print(y_test)

    # 畫出預測值和真實值的關係圖
    plt.plot(y_test, model.predict(x_test), "bo")
    plt.plot([0,50], [0,50], "r-")
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.show()

# 創建兩個模型並進行預測
model_lr = LinearRegression()
model_svr = LinearSVR(random_state=0, tol=1e-3, C=1.2)
predict_value(model_lr, X_train, Y_train, X_test, Y_test)
predict_value(model_svr, X_train, Y_train, X_test, Y_test)