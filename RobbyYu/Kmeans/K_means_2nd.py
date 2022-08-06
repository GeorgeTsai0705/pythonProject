#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

# constant
color = ["b", "g", "r", "y", "m", "c", "k", "lime", "brown", "pink"]

# mode: with reputation / without reputation
mode = "with reputation"


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


def prepare_data(in_frame):
    # schools = in_frame[["School_Name"]]
    df_temp = pd.DataFrame(data=in_frame[["School_Name"]])

    # Indicator: Reputation
    np_list = in_frame[["Academic Reputation", "Employer Reputation "]].fillna(0).to_numpy()
    temp = np.array([x[0] * 0.3 + x[1] * 0.2 for x in np_list]).reshape(1, -1)
    reputation_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=reputation_result[0], columns=["Reputation"])], axis=1)

    # Indicator: Scholar
    np_list = in_frame[["Papers per Faculty", "Citations per Faculty ", "Faculty Staff with PhD"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 0.1 + x[1] * 0.05 + x[2] * 0.05 for x in np_list]).reshape(1, -1)
    scholar_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=scholar_result[0], columns=["Scholar"])], axis=1)

    # Indicator: International
    np_list = in_frame[["International  Faculty", "International Students", "International Research Network",
                        "Inbound Exchange", "Outbound Exchange"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 0.25 + x[1] * 0.025 + x[2] * 0.1 + x[3] * 0.025 + x[4] * 0.025 for x in np_list
                     ]).reshape(1, -1)
    international_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=international_result[0], columns=["International"])], axis=1)

    return df_temp


def main():
    # load ranking data
    df = pd.read_csv("Data/QS_Asia_v2.csv", encoding='unicode_escape')

    # prepare initial data
    df_clean = prepare_data(df)

    # choose mode
    labels = list(df_clean)
    if mode == "with reputation":
        X_new = df_clean[[labels[1], labels[2], labels[3]]].to_numpy()
    else:
        X_new = df_clean[[labels[2], labels[3]]].to_numpy()

    # k-mean training process
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X_new)

    # Search Target
    Target_Index = list(df_clean["School_Name"]).index("National Central University")
    Target_values = X_new[Target_Index]
    Target_group = kmeans.predict([Target_values])[0]

    # Find each group's members
    temp_list = []
    for i in range(len(X_new)):
        temp_list.append([kmeans.predict([X_new[i]])[0], list(df["School_Name"])[i], list(df["Rank_Asia"])[i]])
    df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank_Asia"])
    df_out.sort_values(by=["Group", "School_Name"], inplace=True)
    df_out.to_csv(path_or_buf="C_Result.csv")
    print(Target_group)


    """
    # Final: output file
    df_o.to_csv(path_or_buf="O_Result.csv", header=label_names)
    
    """


if __name__ == '__main__':
    main()
