#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model(dataframe):
    y = dataframe["Faculty Student"].__array__()
    y = y/100
    x = dataframe[["Total faculty staff", "Total students", "ln(Sum)" ,"Ratio"]].__array__()
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled[60:], y[60:], test_size=0.27, random_state=15)
    X_train = np.concatenate((X_train, x_scaled[0:60]), axis=0)
    y_train = np.concatenate((y_train, y[0:60]), axis=0)

    """
    # 不特別取出飽和區做訓練
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=12)
    """
    regr = MLPRegressor(hidden_layer_sizes=(4, 3), max_iter=50000, activation='logistic', early_stopping=True,
                        random_state=29, validation_fraction=0.2, learning_rate="adaptive", solver="lbfgs").fit(X_train, y_train)

    #plt.plot(regr.loss_curve_, 'b.-')
    R2 = regr.score(X_test, y_test)
    print(f"Testing R2 is {R2}.")
    draw_plot(x_scaled, y, regr)

def draw_plot(x, y, regr):
    pred_y = regr.predict(x)
    print(f"All data R2 is {regr.score(x, y)}")
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].plot(y, pred_y, "b.")
    axs[0].plot([0,1], [0,1], "g-", linewidth=1.2)
    axs[0].set_title("Actual vs. Predicted values")
    axs[0].set_ylabel("Predicted values")
    axs[0].set_xlabel("Actual values")
    axs[0].set_ylim([0, 1])
    print(f"Ave Residuals: {np.average(abs(y-pred_y))}")
    axs[1].plot(y, y-pred_y, "r.")
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Predicted Values")
    axs[1].set_title("Residuals vs. Predicted Values")
    axs[1].set_ylim([0,1])
    plt.tight_layout()
    plt.show()

def main():
    # load data from csv file
    df = pd.read_csv('FS_Ratio.csv', encoding='unicode_escape')
    train_model(df)



if __name__ == '__main__':
    main()