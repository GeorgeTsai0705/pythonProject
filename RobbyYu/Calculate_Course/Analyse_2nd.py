#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def main():
    # load data from csv file
    df = pd.read_csv('processed_data/108.csv', encoding='unicode_escape')
    fliter = (df["Semester"] == 1081)
    df = df[fliter]
    Grade_Data = df[["Exam_1","Exam_2","Exam_3","Python_1","Python_2","Python_3"]].dropna().__array__()
    #print(Grade_Data)
    Std_Grade = StandardScaler().fit_transform(X=Grade_Data)
    #print(Std_Grade)

    # Use PCA to reduce Exam & Python dimension
    pca = PCA(n_components=1)
    Exam = pca.fit_transform(Std_Grade[:, 0:3])
    Python = pca.fit_transform(Std_Grade[:, 3:6])
    new = np.concatenate((Exam, Python), axis=1)
    print(new)


    # k = 1~9 做9次kmeans, 並將每次結果的inertia收集在一個list裡
    kmeans_list = [KMeans(n_clusters=k, random_state=123).fit(Std_Grade) for k in range(1, 12)]
    inertias = [model.inertia_ for model in kmeans_list]
    silhouette_scores = [silhouette_score(Std_Grade, model.labels_)for model in kmeans_list[1:]]

    # plot inertias vs K
    plt.figure(1,figsize=(7, 4))
    plt.clf()
    plt.plot(list(range(1, 12)),inertias, 'bo-')
    plt.xlabel("K")
    plt.ylabel("inertias")
    plt.figure(1,figsize=(7, 4))

    # plot silhouette_scores vs K
    plt.figure(2,figsize=(7, 4))
    plt.clf()
    plt.plot(list(range(2, 12)),silhouette_scores, 'bo-')
    plt.xlabel("K")
    plt.ylabel("silhouette_scores")

    plt.show()


    # apply k-mean model
    # predict_result = kmeans.predict(X= Std_Grade)




if __name__ == '__main__':
    main()