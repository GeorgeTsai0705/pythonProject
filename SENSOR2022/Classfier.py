#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def train_model_classifier(df, dc):
    label = list(df)[1:]
    Wavelength = df["Wavelength"].to_numpy()
    df.drop(["Wavelength"], axis=1, inplace=True)
    Wavelength, data = reduce_dimension(Wavelength, df.to_numpy(), 18)
    x_data = data
    y_data = dc.to_numpy()[1][1:].astype(int)

    scaler = StandardScaler().fit(x_data)
    x_scaled = scaler.transform(x_data)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_data, test_size=0.30, random_state=2)
    # kernal function: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

    # Use SVC to classify Result
    classifier = SVC(C=1.3, kernel='linear', gamma='auto', degree=3, probability=True, random_state=11)
    model_process(classifier, X_train, X_test, y_train, y_test, "SVM Classifier", x_scaled, y_data)


    # Use Random Forest
    classifier = RandomForestClassifier(max_depth=4, criterion="entropy", n_estimators=90, random_state=123)
    model_process(classifier, X_train, X_test, y_train, y_test, "Random Forest", x_scaled, y_data)

    # Use MLPClassifier
    classifier = MLPClassifier(hidden_layer_sizes=(3, 2), random_state=12, max_iter=3000,
                               solver="adam", activation="identity")
    model_process(classifier, X_train, X_test, y_train, y_test, "MLPClassifier", x_scaled, y_data)

    # Use Gradient Boosting Classifier
    classifier = GradientBoostingClassifier(n_estimators=120, learning_rate=0.005, max_depth=2,
                                            random_state=11, loss='deviance')
    model_process(classifier, X_train, X_test, y_train, y_test, "Gradient Boosting Classifier", x_scaled, y_data)


def train_model_regression(df, dc):
    label = list(df)[1:]
    Wavelength = df["Wavelength"].to_numpy()
    df.drop(["Wavelength"], axis=1, inplace=True)
    Wavelength, data = reduce_dimension(Wavelength, df.to_numpy(), 10)
    x_data = data
    y_data = dc.to_numpy()[0][1:].astype("float16")

    scaler = StandardScaler().fit(x_data)
    x_scaled = scaler.transform(x_data)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_data, test_size=0.20, random_state=25)

    plt.figure(figsize=(10, 10))

    # Use SVR regression
    regressor = SVR(C=0.8, kernel='linear', gamma='scale', degree=3)
    model_process_rg(regressor, X_train, X_test, y_train, y_test, "SVR Regression", 1)

    # Use Random Forest regression
    regressor = RandomForestRegressor(n_estimators=120, max_depth=3)
    model_process_rg(regressor, X_train, X_test, y_train, y_test, "Random Forest Regression", 2)

    # Use MLPRegressor solver:{‘lbfgs’, ‘sgd’, ‘adam’} activation:{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    regressor = MLPRegressor(hidden_layer_sizes=(4, 2), activation="identity", solver="lbfgs",
                             max_iter=500, random_state=123)
    model_process_rg(regressor, X_train, X_test, y_train, y_test, "MLP Regression", 3)

    # Use GradientBoostingRegressor
    regressor = GradientBoostingRegressor(learning_rate=0.03, n_estimators=160, max_depth=3,
                                          random_state=123, loss="huber")
    model_process_rg(regressor, X_train, X_test, y_train, y_test, "GradientBoosting Regression", 4)

    plt.show()


def model_process_rg(clf, X_train, X_test, y_train, y_test, method, plot_num, x_scaled, y_data):
    start = time.process_time()
    clf.fit(X_train, y_train)
    r2 = clf.score(X_test, y_test)
    end = time.process_time()


    # Output setting
    print("*" * 35)
    print(f"{method} training result")
    print(f"Training Time: {end - start:.3f} s")
    print(f"R2: {r2}")

    # Plot setting
    plt.subplot(2, 2, plot_num)
    plt.plot(y_test, clf.predict(X_test), linestyle='', marker='o', mfc='b', ms='5')
    plt.plot([1, 0], [1, 0], linestyle='--', lw=2, c="gray")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(method)

    return clf


def model_process(clf, X_train, X_test, y_train, y_test, method, x_scaled, y_data):
    start = time.process_time()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    result = precision_recall_fscore_support(y_test, clf.predict(X_test), average='macro')
    end = time.process_time()
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    print("*" * 35)
    print(f"{method} training result")
    print(f"Training Time: {end - start:.3f} s")
    print(
        f"Accuracy: {accuracy * 100:.1f}, Precision: {result[0] * 100:.1f}, Recall: {result[1] * 100:.1f}, F1:{result[2] * 100:.1f}")
    print("Confusion Matrix: ")
    print(conf_matrix)

    cross_val = cross_val_score(clf, x_scaled, y_data, cv=5)
    print(f"{method} cross: {cross_val}, avg: {np.average(cross_val)}")
    return clf


def PCA_preprocess(data: object) -> object:
    pca = PCA(n_components=2)
    data = pca.fit_transform(X=data)
    return np.array(data)


def reduce_dimension(lambada: object, data: object, group_num: object) -> object:
    loop_num = len(lambada) // group_num
    new_lambada = []
    for i in range(loop_num):
        new_lambada.append(np.average(lambada[i * group_num:(i + 1) * group_num]))
    new_data = []
    for sample in data.transpose():
        temp_data = []
        for i in range(loop_num):
            temp_data.append(np.average(sample[i * group_num:(i + 1) * group_num]))
        new_data.append(temp_data)
    return np.array(new_lambada), np.array(new_data)


def main():
    # load data from csv file
    df = pd.read_csv('Temp_Alld_tline.csv', encoding='unicode_escape')
    dc = pd.read_csv('cPassResult.csv', encoding='unicode_escape')
    train_model_classifier(df, dc)
    # train_model_regression(df, dc)


if __name__ == '__main__':
    main()
