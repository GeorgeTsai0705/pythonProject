#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# constant
color = ["b", "g", "r", "y", "m", "c", "k", "lime", "brown", "pink"]

# load ranking data
df = pd.read_csv("Data/QS_Asia_v2.csv", encoding='unicode_escape')
df = df.fillna(0)
title = list(df)

X = df[[title[6], title[7], title[8], title[9], title[10], title[11]]]
X_new = X.to_numpy()

for num in [8]:
    kmeans = KMeans(n_clusters=num, random_state=0).fit(X_new)

    # Search Target
    Target_Index = list(df["School_Name"]).index("National Central University")
    Target_values = X_new[Target_Index]
    Target_group = kmeans.predict([Target_values])[0]
    print(Target_values)

count = 0
for ele in X_new:
    count += 1
    if kmeans.predict([ele])[0] == Target_group:
        print(list(df["School_Name"])[count])

"""
    for ele in X_new:
        plt.scatter(ele[0], ele[1], c=color[kmeans.predict([ele])[0]], marker='.', s=4)

    plt.scatter(Target_values[0], Target_values[1], c=color[kmeans.predict([Target_values])[0]], marker='X', s=20)
    print(kmeans.n_iter_)
    plt.xlabel(title[6])
    plt.ylabel(title[7])
    plt.show()
"""