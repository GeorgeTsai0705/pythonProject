#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model(dataframe):
    y = dataframe["Faculty Student"].__array__()
    x = dataframe[["Total students", "ln(Sum)", "ln(Ratio)"]].__array__()
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=12)
    regr = MLPRegressor(hidden_layer_sizes=(4, 3), max_iter=50000, activation='logistic', early_stopping=True,
                        random_state=22, validation_fraction=0.2, learning_rate="adaptive").fit(X_train, y_train)
    plt.plot(regr.loss_curve_, 'b.-')
    R2 = regr.score(X_test, y_test)
    print(R2)
    draw_plot(x_scaled, y, regr)

def draw_plot(x, y, regr):
    pred_y = regr.predict(x)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    axs[0].plot(y, pred_y, "b.")
    axs[0].set_title("Actual vs. Predicted values")
    axs[0].set_ylim([0,100])
    axs[1].plot(y, y-pred_y, "r.")
    axs[1].set_title("Residuals vs. Predicted Values")
    axs[1].set_ylim([0,100])

    """
    
    plt.subplot(212)
    plt.plot(y, pred_y-y)
    fig.suptitle("Plotting cross-validated predictions")
    """
    plt.tight_layout()
    plt.show()

def main():
    # load data from csv file
    df = pd.read_csv('FS_Ratio.csv', encoding='unicode_escape')
    train_model(df)



if __name__ == '__main__':
    main()