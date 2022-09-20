#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt

# constant
color = ["b", "g", "r", "y", "m", "c", "k", "lime", "brown", "pink"]
target_school = "National Central University"

# mode: with reputation / without reputation
mode = "with reputation"


def prepare_asia_data(in_frame):
    # schools = in_frame[["School_Name"]]
    df_temp = pd.DataFrame(data=in_frame[["School_Name", "Overall"]])

    # Indicator: Reputation
    np_list = in_frame[["Academic Reputation", "Employer Reputation "]].fillna(0).to_numpy()
    # Special Weight
    # temp = np.array([x[0] * 0.1 + x[1] * 0.1 for x in np_list]).reshape(1, -1)
    # Original Weight
    temp = np.array([x[0] * 0.3 + x[1] * 0.2 for x in np_list]).reshape(1, -1)
    reputation_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=reputation_result[0], columns=["Reputation"])], axis=1)

    # Indicator: Scholar
    np_list = in_frame[["Papers per Faculty", "Citations per Faculty ", "Faculty Staff with PhD", "Faculty Student"]].fillna(0).to_numpy()
    # Special Weight
    # temp = np.array([x[0] * 0.1 + x[1] * 0.1 + x[2] * 0.1 + x[3] * 0.1for x in np_list]).reshape(1, -1)
    # Original Weight
    temp = np.array([x[0] * 0.1 + x[1] * 0.05 + x[2] * 0.05 + x[3] * 0.1 for x in np_list]).reshape(1, -1)
    scholar_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=scholar_result[0], columns=["Scholar"])], axis=1)

    # Indicator: International
    np_list = in_frame[["International  Faculty", "International Students", "International Research Network",
                        "Inbound Exchange", "Outbound Exchange"]].fillna(0).to_numpy()
    # Special Weight
    # temp = np.array([x[0] * 1 + x[1] * 1 + x[2] * 1 + x[3] * 1 + x[4] * 1 for x in np_list]).reshape(1, -1)
    # Original Weight
    # temp = np.array([x[0] * 0.25 + x[1] * 0.025 + x[2] * 0.1 + x[3] * 0.025 + x[4] * 0.025 for x in np_list
    #                      ]).reshape(1, -1)
    international_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=international_result[0], columns=["International"])], axis=1)

    return df_temp


def qs_asia_fn(df):
    # prepare initial data
    df_clean = prepare_asia_data(df)

    # choose mode
    labels = list(df_clean)
    if mode == "with reputation":
        X_new = df_clean[[labels[2], labels[3], labels[4]]].to_numpy()
    else:
        X_new = df_clean[[labels[3], labels[4]]].to_numpy()

    # k-mean training process
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X_new)

    # Search Target and print the group of target
    Target_Index = list(df_clean["School_Name"]).index(target_school)
    Target_values = X_new[Target_Index]
    Target_group = kmeans.predict([Target_values])[0]
    print(Target_group)

    # Find each group's members
    temp_list = []
    for i in range(len(X_new)):
        temp_list.append([kmeans.predict([X_new[i]])[0], list(df_clean["School_Name"])[i], list(df["Rank_Asia"])[i],
                          list(df_clean["Overall"])[i]])

    # Prepare output file for the k-mean result
    df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank_Asia", 'Overall'])
    df_out.sort_values(by=["Group", "School_Name"], inplace=True)
    df_out.to_csv(path_or_buf="C_Result.csv")

    # Calculate statistic parameters
    statistic_data = []
    for i in range(9):
        mask = df_out["Group"] == i
        statistic_data.append(
            [i, df_out[mask]["Rank_Asia"].mean(0), df_out[mask]["Rank_Asia"].std(0), df_out[mask].count(0).values[0]])
    pd.DataFrame(data=statistic_data, columns=["Group", "Mean", "Std", 'Count']).to_csv(path_or_buf="S_Result.csv")


def prepare_the_asia_data(in_frame):
    # schools = in_frame[["School_Name"]]
    df_temp = pd.DataFrame(data=in_frame[["School_Name", "Overall"]])

    # Indicator: Reputation
    np_list = in_frame[["Industry income", "Research"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 1 + x[1] * 1 for x in np_list]).reshape(1, -1)
    # Original Weight
    # temp = np.array([x[0] * 0.075 + x[1] * 0.3 for x in np_list]).reshape(1, -1)
    reputation_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=reputation_result[0], columns=["Reputation"])], axis=1)

    # Indicator: Scholar
    np_list = in_frame[["Citations", "Teaching"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 1 + x[1] * 1 for x in np_list]).reshape(1, -1)
    # Original Weight
    # temp = np.array([x[0] * 0.3 + x[1] * 0.25 for x in np_list]).reshape(1, -1)
    scholar_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=scholar_result[0], columns=["Scholar"])], axis=1)

    # Indicator: International
    np_list = in_frame[["International outlook"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 1 for x in np_list]).reshape(1, -1)
    # Original Weight
    # temp = np.array([x[0] * 0.08 for x in np_list]).reshape(1, -1)
    international_result = Normalizer(norm='max').fit_transform(X=temp)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=international_result[0], columns=["International"])], axis=1)

    return df_temp


def the_asia_fn(df):
    # prepare initial data
    df_clean = prepare_the_asia_data(df)

    # choose mode
    labels = list(df_clean)
    if mode == "with reputation":
        X_new = df_clean[[labels[2], labels[3], labels[4]]].to_numpy()
    else:
        X_new = df_clean[[labels[3], labels[4]]].to_numpy()

    # k-mean training process
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X_new)

    # Search Target and print the group of target
    Target_Index = list(df_clean["School_Name"]).index(target_school)
    Target_values = X_new[Target_Index]
    Target_group = kmeans.predict([Target_values])[0]
    print(Target_group)

    # Find each group's members
    temp_list = []
    for i in range(len(X_new)):
        temp_list.append([kmeans.predict([X_new[i]])[0], list(df_clean["School_Name"])[i], list(df["Rank_Asia"])[i],
                          list(df_clean["Overall"])[i]])

    # Prepare output file for the k-mean result
    df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank_Asia", 'Overall'])
    df_out.sort_values(by=["Group", "School_Name"], inplace=True)
    df_out.to_csv(path_or_buf="C_Result.csv")

    # Calculate statistic parameters
    statistic_data = []
    for i in range(9):
        mask = df_out["Group"] == i
        statistic_data.append(
            [i, df_out[mask]["Rank_Asia"].mean(0), df_out[mask]["Rank_Asia"].std(0), df_out[mask].count(0).values[0]])
    pd.DataFrame(data=statistic_data, columns=["Group", "Mean", "Std", 'Count']).to_csv(path_or_buf="S_Result.csv")


def special_case(in_frame):
    np_list = in_frame[["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]].fillna(0).to_numpy()
    result = StandardScaler().fit_transform(X=np_list)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(result)
    for ele in result:
        K = kmeans.predict([ele])[0]
        print(K)


def main():
    #df = pd.read_csv("Data/Taiwan_school.csv", encoding='unicode_escape')
    #special_case(df)

    # load ranking data
    data_type = "QS"  # THE/QS
    df = pd.read_csv(f'Data/{data_type}_Asia.csv', encoding='unicode_escape')

    if data_type == "QS":
        # perform qs result analyse
        qs_asia_fn(df)
    elif data_type == "THE":
        # perform the result analyse
        the_asia_fn(df)


if __name__ == '__main__':
    main()
