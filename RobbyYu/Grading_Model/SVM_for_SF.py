#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model(dataframe):
    y = dataframe["Faculty Student"].__array__()
    x = dataframe[["Total students", "Total faculty staff", "Ratio", "Sum"]].__array__()
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12)
    print(X_train)

def main():
    # load data from csv file
    df = pd.read_csv('FS_Ratio.csv', encoding='unicode_escape')
    train_model(df)



if __name__ == '__main__':
    main()