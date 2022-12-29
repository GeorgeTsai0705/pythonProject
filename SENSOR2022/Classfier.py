#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support


def train_model(df, dc):
    label = list(df)[1:]
    Wavelength = df["Wavelength"].to_numpy()
    df.drop(["Wavelength"], axis=1, inplace=True)
    Wavelength, data = reduce_dimension(Wavelength, df.to_numpy(), 5)
    x_data = data
    y_data = dc.to_numpy()[1][1:].astype(int)

    scaler = StandardScaler().fit(x_data)
    x_scaled = scaler.transform(x_data)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_data, test_size=0.25, random_state=15)
    # kernal function: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    classifier = SVC(kernel='poly', gamma='auto', random_state= 12)

    start = time.process_time_ns()
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    result = precision_recall_fscore_support(y_test, classifier.predict(X_test), average='macro')
    end = time.process_time_ns()
    print(f"Training Time: {end-start:.3f} ms")
    print(f"Accuracy: {accuracy}, Precision: {result[0]}, Recall: {result[1]}, F1:{result[2]}")

def PCA_preprocess(data: object) -> object:
    pca = PCA(n_components=2)
    data = pca.fit_transform(X = data)
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
    train_model(df, dc)


if __name__ == '__main__':
    main()
