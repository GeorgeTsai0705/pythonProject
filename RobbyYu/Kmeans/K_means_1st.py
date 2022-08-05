#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# constant
color = ["b", "g", "r", "y", "m", "c", "k", "lime", "brown", "pink"]


def sort_function(out_frame, in_frame):
    college_name = list(in_frame["School_Name"])
    labels = list(out_frame)
    count_result = []
    for name in college_name:
        c = 0
        for na in labels:
            if name in list(out_frame[na]):
                c += 1
        count_result.append(c)

    return count_result


def main():
    # load ranking data
    df = pd.read_csv("Data/QS_Asia_v2.csv", encoding='unicode_escape')
    title = list(df)

    # prepare output file
    df_o = pd.DataFrame()
    label_names = []

    # prepare index series
    index_list = []
    for i in range(4, 20):
        for j in range(i + 1, 20):
            index_list.append((i, j))
    print(index_list)

    for element in index_list:

        X = df[["School_Name", title[element[0]], title[element[1]]]]
        X = X.dropna(subset=[title[element[0]], title[element[1]]])
        X_new = X[[title[element[0]], title[element[1]]]].to_numpy()

        # KMeans train
        num = 9
        kmeans = KMeans(n_clusters=num, random_state=0).fit(X_new)

        # Search Target
        Target_Index = list(X["School_Name"]).index("National Central University")
        Target_values = X_new[Target_Index]
        Target_group = kmeans.predict([Target_values])[0]

        temp_list = []

        for i in range(len(X_new)):
            if kmeans.predict([X_new[i]])[0] == Target_group:
                temp_list.append(list(X["School_Name"])[i])

        label_names.append(title[element[0]] + " vs " + title[element[1]])
        df_temp = pd.DataFrame(data=temp_list, columns= [title[element[0]] + " vs " + title[element[1]]])
        df_o = pd.concat([df_o, df_temp], axis=1)

        print("Finish : " + title[element[0]] + " vs " + title[element[1]])

    # sort result
    count_result = sort_function(df_o, X)
    df_c = pd.DataFrame(index=X["School_Name"], data=count_result)

    # Final: output file
    df_o.to_csv(path_or_buf="O_Result.csv", header=label_names)
    df_c.to_csv(path_or_buf="C_Result.csv")



if __name__ == '__main__':
    main()
