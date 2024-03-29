#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# constant
color = ["b", "g", "r", "y", "m", "c", "k", "lime", "brown", "pink"]
target_school = "National Central University"

np.set_printoptions(precision=3, suppress=False)
# mode: A(Reputation vs Scholar)；B(Reputation vs International)；C(Scholar vs International)
mode = "A"
data_type = "SEA"  # THE/QS/SEA
k_cluster_num = 5

def prepare_asia_data(in_frame):
    # schools = in_frame[["School_Name"]]
    df_temp = pd.DataFrame(data=in_frame[["School_Name", "Overall"]])

    # Indicator: Reputation
    np_list = in_frame[["Academic Reputation", "Employer Reputation "]].fillna(0).to_numpy()
    # Special Weight
    # temp = np.array([x[0] * 0.1 + x[1] * 0.1 for x in np_list]).reshape(-1, 1)
    # Original Weight
    temp = np.array([x[0] * 0.3 + x[1] * 0.2 for x in np_list]).reshape(-1, 1)
    reputation_result = StandardScaler().fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=reputation_result[0], columns=["Reputation"])], axis=1)

    # Indicator: Scholar
    np_list = in_frame[["Papers per Faculty", "Citations per Faculty ", "Faculty Staff with PhD", "Faculty Student"]].fillna(0).to_numpy()
    # Special Weight
    # temp = np.array([x[0] * 0.1 + x[1] * 0.1 + x[2] * 0.1 + x[3] * 0.1for x in np_list]).reshape(-1, 1)
    # Original Weight
    temp = np.array([x[0] * 0.1 + x[1] * 0.05 + x[2] * 0.05 + x[3] * 0.1 for x in np_list]).reshape(-1, 1)
    scholar_result = StandardScaler().fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=scholar_result[0], columns=["Scholar"])], axis=1)

    # Indicator: International
    np_list = in_frame[["International  Faculty", "International Students", "International Research Network",
                        "Inbound Exchange", "Outbound Exchange"]].fillna(0).to_numpy()
    # Special Weight
    # temp = np.array([x[0] * 1 + x[1] * 1 + x[2] * 1 + x[3] * 1 + x[4] * 1 for x in np_list]).reshape(-1, 1)
    # Original Weight
    temp = np.array([x[0] * 0.25 + x[1] * 0.025 + x[2] * 0.1 + x[3] * 0.025 + x[4] * 0.025 for x in np_list]).reshape(-1, 1)
    international_result = StandardScaler().fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=international_result[0], columns=["International"])], axis=1)

    return df_temp


def qs_asia_fn(df):
    # prepare initial data
    df_clean = prepare_asia_data(df)

    # choose mode
    labels = list(df_clean)
    if mode == "A":
        X_new = df_clean[[labels[2], labels[3]]].to_numpy()
    elif mode == "B":
        X_new = df_clean[[labels[2], labels[4]]].to_numpy()
    else:
        X_new = df_clean[[labels[3], labels[4]]].to_numpy()

    # k-mean training process
    kmeans = KMeans(n_clusters=k_cluster_num, random_state=0).fit(X_new)

    # Search Target and print the group of target
    Target_Index = list(df_clean["School_Name"]).index(target_school)
    Target_values = X_new[Target_Index]
    Target_group = kmeans.predict([Target_values])[0]
    print(f'{target_school} is in Group {Target_group}')

    # Find each group's members
    temp_list = []
    for i in range(len(X_new)):
        temp_list.append([kmeans.predict([X_new[i]])[0]+1, list(df_clean["School_Name"])[i], list(df["Rank_Asia"])[i],
                          X_new[i][0], X_new[i][1]])

    # Prepare output file for the k-mean result
    #df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank_Asia", 'Overall'])
    df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank_Asia", 'A', 'B'])
    df_out.sort_values(by=["Group", "School_Name"], inplace=True)
    df_out.to_csv(path_or_buf="C_Result.csv")

    # Draw Visualize Graph
    KMeanVisualize(data=X_new, classifier=kmeans, mode= mode)
    plt.show()

    # Calculate statistic parameters
    statistic_data = []
    for i in range(k_cluster_num):
        mask = df_out["Group"] == i+1
        statistic_data.append(
            [i, df_out[mask]["Rank_Asia"].mean(0), df_out[mask]["Rank_Asia"].std(0), df_out[mask].count(0).values[0]])
    pd.DataFrame(data=statistic_data, columns=["Group", "Mean", "Std", 'Count']).to_csv(path_or_buf="S_Result.csv")


def prepare_the_asia_data(in_frame):
    # schools = in_frame[["School_Name"]]
    df_temp = pd.DataFrame(data=in_frame[["School_Name", "Overall"]])

    # Indicator: Reputation
    np_list = in_frame[["Teaching", "Research"]].fillna(0).to_numpy()
    # Equal Weight
    # temp = np.array([x[0] * 0.4 + x[1] * 0.5 for x in np_list]).reshape(1, -1)
    # Original Weight
    temp = np.array([x[0] * 0.4 * 0.1 + x[1] * 0.5 * 0.15 for x in np_list]).reshape(-1, 1)
    reputation_result = StandardScaler().fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=reputation_result[0], columns=["Reputation"])], axis=1)

    # Indicator: Scholar
    np_list = in_frame[["Citations", "Teaching", "Research", "Industry income"]].fillna(0).to_numpy()
    # Equal Weight
    # temp = np.array([x[0] * 1 + x[1] * 1 for x in np_list]).reshape(1, -1)
    # Original Weight
    temp = np.array([x[0] * 0.3 + x[1] * 0.6 * 0.15 + x[2] * 0.5 * 0.15 + x[3] * 0.075 for x in np_list]).reshape(-1, 1)
    scholar_result = StandardScaler().fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=scholar_result[0], columns=["Scholar"])], axis=1)

    # Indicator: International
    np_list = in_frame[["International outlook"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 1 for x in np_list]).reshape(-1, 1)
    # Original Weight
    # temp = np.array([x[0] * 0.08 for x in np_list]).reshape(1, -1)
    international_result = StandardScaler().fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=international_result[0], columns=["International"])], axis=1)

    return df_temp


def the_asia_fn(df):
    # prepare initial data
    df_clean = prepare_the_asia_data(df)

    # choose mode
    labels = list(df_clean)
    if mode == "A":
        X_new = df_clean[[labels[2], labels[3]]].to_numpy()
    elif mode == "B":
        X_new = df_clean[[labels[2], labels[4]]].to_numpy()
    else:
        X_new = df_clean[[labels[3], labels[4]]].to_numpy()

    # k-mean training process
    kmeans = KMeans(n_clusters=k_cluster_num, random_state=0).fit(X_new)

    # Search Target and print the group of target
    Target_Index = list(df_clean["School_Name"]).index(target_school)
    Target_values = X_new[Target_Index]
    Target_group = kmeans.predict([Target_values])[0]
    print(f'{target_school} is in Group {Target_group}')

    # Find each group's members
    temp_list = []
    for i in range(len(X_new)):
        temp_list.append([kmeans.predict([X_new[i]])[0], list(df_clean["School_Name"])[i], list(df["Rank_Asia"])[i],
                          X_new[i][0], X_new[i][1]])

    # Prepare output file for the k-mean result
    df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank_Asia", 'A', 'B'])
    df_out.sort_values(by=["Group", "School_Name"], inplace=True)
    df_out.to_csv(path_or_buf="C_Result.csv")

    # Draw Visualize Graph
    KMeanVisualize(data=X_new, classifier=kmeans, mode= mode)
    plt.show()

    # Calculate statistic parameters
    statistic_data = []
    for i in range(k_cluster_num):
        mask = df_out["Group"] == i
        statistic_data.append(
            [i, df_out[mask]["Rank_Asia"].mean(0), df_out[mask]["Rank_Asia"].std(0), df_out[mask].count(0).values[0]])
    pd.DataFrame(data=statistic_data, columns=["Group", "Mean", "Std", 'Count']).to_csv(path_or_buf="S_Result.csv")


def prepare_SEasia_data(in_frame, TW_frame):
    df_temp = pd.DataFrame(data=in_frame[["School_Name", "Location", "Overall"]])
    df_TW = pd.DataFrame(data=TW_frame[["School_Name", "Location", "Overall"]])
    Std = StandardScaler()

    # Indicator: Reputation
    np_list = in_frame[["Academic Reputation", "Employer Reputation "]].fillna(0).to_numpy()
    # Normal Weight
    temp = np.array([x[0] * 0.5 + x[1] * 0.5 for x in np_list]).reshape(-1, 1)
    reputation_result = Std.fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=reputation_result[0], columns=["Reputation"])], axis=1)

    # Process StandardScaler() for Taiwanese Data in Reputation
    np_list = TW_frame[["Academic Reputation", "Employer Reputation "]].fillna(0).to_numpy()
    temp = np.array([x[0] * 0.5 + x[1] * 0.5 for x in np_list]).reshape(-1, 1)
    TW_reputation_result = Std.transform(X=temp).reshape(1, -1)
    df_TW = pd.concat([df_TW, pd.DataFrame(data=TW_reputation_result[0], columns=["Reputation"])], axis=1)

    # Indicator: Scholar
    np_list = in_frame[["Faculty Student", "Citations per Faculty "]].fillna(0).to_numpy()
    # Normal Weight
    temp = np.array([x[0] * 0.5 + x[1] * 0.5 for x in np_list]).reshape(-1, 1)
    scholar_result = Std.fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=scholar_result[0], columns=["Scholar"])], axis=1)

    # Process StandardScaler() for Taiwanese Data in Scholar
    np_list = TW_frame[["Faculty Student", "Citations per Faculty "]].fillna(0).to_numpy()
    temp = np.array([x[0] * 0.5 + x[1] * 0.5 for x in np_list]).reshape(-1, 1)
    TW_scholar_result = Std.transform(X=temp).reshape(1, -1)
    df_TW = pd.concat([df_TW, pd.DataFrame(data=TW_scholar_result[0], columns=["Scholar"])], axis=1)


    # Indicator: International
    np_list = in_frame[["International Faculty ", "International Students"]].fillna(0).to_numpy()
    # Normal Weight
    temp = np.array([x[0] * 0.5 + x[1] * 0.5 for x in np_list]).reshape(-1, 1)
    international_result = Std.fit_transform(X=temp).reshape(1, -1)
    df_temp = pd.concat([df_temp, pd.DataFrame(data=international_result[0], columns=["International"])], axis=1)

    # Process StandardScaler() for Taiwanese Data in International
    np_list = TW_frame[["International Faculty ", "International Students"]].fillna(0).to_numpy()
    temp = np.array([x[0] * 0.5 + x[1] * 0.5 for x in np_list]).reshape(-1, 1)
    TW_international_result = Std.transform(X=temp).reshape(1, -1)
    df_TW = pd.concat([df_TW, pd.DataFrame(data=TW_international_result[0], columns=["International"])], axis=1)
    df_TW.to_csv(path_or_buf="TW_Result.csv")

    return df_temp, df_TW

def SEasia_fn(df):
    # load Taiwanese University Data
    dTW = pd.read_csv('Data/SEA_Asia(TW)_2023.csv', encoding='unicode_escape')

    # prepare initial data
    df_clean, dTW_clean = prepare_SEasia_data(df, dTW)

    # choose mode
    labels = list(df_clean)
    for loop in range(3):
        if loop == 0:
            X_new =  df_clean[[labels[3], labels[4]]].to_numpy()
        elif loop == 1:
            X_new = df_clean[[labels[3], labels[5]]].to_numpy()
        else:
            X_new = df_clean[[labels[4], labels[5]]].to_numpy()


        # 找出適當的分群數量
        kmeans_list = [KMeans(n_clusters=k, random_state=123).fit(X_new) for k in range(3, 8)]
        inertias = [model.inertia_ for model in kmeans_list]

        """
        # plot inertias vs K 
        plt.figure(1, figsize=(7, 4))
        plt.clf()
        plt.plot(list(range(3, 8)), inertias, 'bo-')
        plt.xlabel("K")
        plt.ylabel("inertias")
        plt.figure(1, figsize=(7, 4))
        plt.show()
        """


        # k-mean training process
        kmeans = KMeans(n_clusters=k_cluster_num, random_state=0).fit(X_new)

        # Find each group's members
        temp_list = []
        for i in range(len(X_new)):
            temp_list.append(
                [kmeans.predict([X_new[i]])[0] + 1, list(df_clean["School_Name"])[i], list(df["Rank"])[i],
                 X_new[i][0], X_new[i][1]])

        # Prepare output file for the k-mean result
        df_out = pd.DataFrame(data=temp_list, columns=["Group", "School_Name", "Rank", 'A', 'B'])
        df_out.sort_values(by=["Group", "School_Name"], inplace=True)
        df_out.to_csv(path_or_buf="C_Result.csv")

        # Draw Visualize Graph
        KMeanVisualize(data=X_new, classifier=kmeans, mode=loop)
        if loop == 0:
            plt.plot(dTW_clean["Reputation"].to_numpy(), dTW_clean["Scholar"].to_numpy(), 'rx', markersize=6)
        elif loop == 1:
            plt.plot(dTW_clean["Reputation"].to_numpy(), dTW_clean["International"].to_numpy(), 'rx', markersize=6)
        else:
            plt.plot(dTW_clean["Scholar"].to_numpy(), dTW_clean["International"].to_numpy(), 'rx', markersize=6)

        plt.show()


def special_case(in_frame):
    np_list = in_frame[["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]].fillna(0).to_numpy()
    result = StandardScaler().fit_transform(X=np_list)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(result)
    for ele in result:
        K = kmeans.predict([ele])[0]
        print(K)

def KMeanVisualize(data, classifier, mode):
    h = 0.01

    # Plot the decision boundary. For that, we will assign a color to each
    #x_min, x_max = data[:, 0].min() - 1,  data[:, 0].max() + 1
    x_min = -3; x_max = 3;y_min = -3; y_max = 3

    #y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max,h))

    # Obtain labels for each point in mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(7, 7))
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    plt.plot(data[:,0],data[:,1],"k.", markersize=5)
    centroids = classifier.cluster_centers_

    for ele in centroids:
        plt.text(
            ele[0],
            ele[1],
            s = str(classifier.predict([ele])+1),
            c = 'white'
        )

    if mode == "A" or mode == 0:
        plt.xlabel(xlabel="Reputation", size='x-large');plt.ylabel(ylabel="Scholar",size='x-large');
    elif mode == "B" or mode == 1:
        plt.xlabel(xlabel="Reputation", size='x-large');plt.ylabel(ylabel="International", size='x-large');
    else:
        plt.xlabel(xlabel="Scholar", size='x-large');plt.ylabel(ylabel="International", size='x-large');
    #plt.xticks()
    plt.yticks()
    return plt


def main():
    #df = pd.read_csv("Data/Taiwan_school.csv", encoding='unicode_escape')
    #special_case(df)

    # load ranking data
    df = pd.read_csv(f'Data/{data_type}_Asia.csv', encoding='unicode_escape')

    if data_type == "QS":
        # perform qs result analyse
        qs_asia_fn(df)
    elif data_type == "THE":
        # perform the result analyse
        the_asia_fn(df)
    elif data_type == "SEA":
        SEasia_fn(df)


if __name__ == '__main__':
    main()
