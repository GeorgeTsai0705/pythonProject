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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, r2_score, mean_absolute_error


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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, r2_score, mean_absolute_error

# Global constants
TEST_SIZE = 0.3
RANDOM_STATE = 7
GROUP_NUM = 50

def load_data(file_path):
    return pd.read_csv(file_path, encoding='unicode_escape')

def process_data(df):
    Wavelength = df["Wavelength"].to_numpy()
    df.drop(["Wavelength"], axis=1, inplace=True)
    Wavelength, data = reduce_dimension(Wavelength, df.to_numpy(), GROUP_NUM)
    return Wavelength, data

def scale_data(x_data):
    scaler = StandardScaler().fit(x_data)
    return scaler.transform(x_data)

def reduce_dimension(lambada, data, group_num):
    loop_num = len(lambada) // group_num
    new_lambada = [np.average(lambada[i * group_num:(i + 1) * group_num]) for i in range(loop_num)]
    new_data = [[np.average(sample[i * group_num:(i + 1) * group_num]) for i in range(loop_num)] for sample in data.transpose()]
    return np.array(new_lambada), np.array(new_data)

def percent2IU(a):
    return np.exp(a*3.66)*6.0175 if not isinstance(a, list) else [np.exp(x*3.66)*6.0175 for x in a]

def model_process(clf, X_train, X_test, y_train, y_test, method, x_scaled, y_data):
    start = time.process_time()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, clf.predict(X_test), average='macro')
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))

    print(f"{method} - Training Time: {time.process_time() - start:.3f} s, "
          f"Accuracy: {accuracy * 100:.1f}, "
          f"Precision: {precision * 100:.1f}, "
          f"Recall: {recall * 100:.1f}, "
          f"F1:{f1 * 100:.1f}")
    print("Confusion Matrix: ")
    print(conf_matrix)
    print(f"{method} - Cross-Validation Score: {np.average(cross_val_score(clf, x_scaled, y_data, cv=5))}")

    return clf


def model_process_rg(rg, X_train, X_test, y_train, y_test, method, plot_num, y_max, y_min):
    start = time.process_time()
    rg.fit(X_train, y_train)
    r2 = rg.score(X_test, y_test)
    end = time.process_time()

    # Output setting
    print("*" * 35)
    print(f"{method} training result")
    print(f"Training Time: {end - start:.3f} s")
    print(f"R2: {r2}, MAE: {mean_absolute_error(percent2IU(y_test), percent2IU(rg.predict(X_test)))}")

    # Check Low & High conc. performance
    L_X=[]; L_y=[]; H_X=[];H_y=[]

    for i in range(len(y_test)):
        if y_test[i] < 0.35:
            L_X.append(X_test[i])
            L_y.append(y_test[i])
        else:
            H_X.append(X_test[i])
            H_y.append(y_test[i])
    print(f"Low R2: {mean_absolute_error(percent2IU(L_y), percent2IU(rg.predict(L_X))):.3f}")
    print(f"High R2: {mean_absolute_error(percent2IU(H_y), percent2IU(rg.predict(H_X))):.3f}")

    # Plot setting
    plt.subplot(2, 2, plot_num)
    plt.plot(percent2IU(y_test),  percent2IU(rg.predict(X_test)), linestyle='', marker='o', mfc='b', ms='4')
    plt.plot([200, 0], [200, 0], linestyle='--', lw=2, c="gray")
    plt.xlim([0, 220])
    plt.ylim([0, 220])
    plt.xlabel("Measured neutralizing antibody (IU/mL)")
    plt.ylabel("Predicted neutralizing antibody (IU/mL)")
    plt.fill_between(x=[0, 200], y1=[20, 220], y2=[-20, 180], color='C0', alpha=0.1, label='Area')
    plt.title(method)

    output = pd.DataFrame(data= [percent2IU(y_test), percent2IU(rg.predict(X_test))])
    output.to_csv(f"{method}.csv")

    return rg


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

def percent2IU(a):
    if type(a) == type([1]):
        o = [np.exp(x*3.66)*6.0175 for x in a]
        return o
    else:
        return np.exp(a*3.66)*6.0175

def special_reg(df, dc):
    # Train classifier first
    Wavelength = df["Wavelength"].to_numpy()
    df.drop(["Wavelength"], axis=1, inplace=True)
    Wavelength, data = reduce_dimension(Wavelength, df.to_numpy(), 50)
    x_data = data
    y_data_c = dc.to_numpy()[3][1:].astype(int)

    scaler = StandardScaler().fit(x_data)
    x_scaled = scaler.transform(x_data)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_data_c, test_size=0.3, random_state=10, stratify=y_data_c)
    # kernal function: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # Use SVC to classify Result
    classifier = SVC(C=1.0, kernel='linear', gamma='auto', degree=4, probability=True, random_state=1)
    clf = model_process(classifier, X_train, X_test, y_train, y_test, "SVM Classifier", x_scaled, y_data_c)

    # Try fit reg for MPOS data
    newX = [];
    newY = []
    y_data = dc.to_numpy()[0][1:].astype("float16")
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_data, test_size=0.3, random_state=7, stratify=y_data_c)

    # Use Random Forest regression
    regressor = RandomForestRegressor(n_estimators=20, max_depth=5)
    regressor.fit(X_train, y_train)
    print(regressor.score(X_test, y_test))

    # Use SVR regression
    regressor2 = SVR(C=0.3, kernel='linear', gamma='scale', degree=2, tol=0.0001)
    regressor2.fit(X_train, y_train)
    r2 = regressor2.score(X_test, y_test)

    print(f"Original R2: {r2}, MAE: {mean_absolute_error(percent2IU(y_test), percent2IU(regressor.predict(X_test)))}")

    y_result = []
    # use classifier to distinguish HPOS & NEG
    for i in range(len(y_test)):
        if clf.predict([X_test[i]]) == 0:
            y_result.append(regressor2.predict([X_test[i]]))
        else:
            y_result.append(regressor.predict([X_test[i]]))
    print(f"improve: {r2_score(y_test, y_result)}, MAE: {mean_absolute_error(percent2IU(y_test), percent2IU(y_result))}")

    # Plot setting
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 1, 1)
    plt.plot(percent2IU(y_test), percent2IU(regressor.predict(X_test)), linestyle='', marker='o', mfc='b', ms='3')
    plt.plot([200, 0], [200, 0], linestyle='--', lw=2, c="gray")
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    plt.fill_between(x=[0, 200], y1=[20, 220], y2=[-20, 180], color='C0', alpha=0.1, label='Area')
    plt.title('Original')

    plt.subplot(2, 1, 2)
    plt.plot(percent2IU(y_test), percent2IU(y_result), linestyle='', marker='o', mfc='b', ms='3')
    plt.plot([200, 0], [200, 0], linestyle='--', lw=2, c="gray")
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.xlabel("Measured neutralizing antibody (IU/mL)")
    plt.ylabel("Predicted neutralizing antibody (IU/mL)")
    plt.fill_between(x=[0, 200], y1=[20, 220], y2=[-20, 180], color='C0', alpha=0.1, label='Area')
    plt.title('Improvement')
    plt.show()

    output = pd.DataFrame(data= [percent2IU(y_test), percent2IU(y_result)])
    output.to_csv("Output.csv")

def main():
    # load data from csv file
    df = pd.read_csv('Temp_Alld_tline.csv', encoding='unicode_escape')
    dc = pd.read_csv('cPassResult.csv', encoding='unicode_escape')
    trained_clf, newX, newY = train_model_classifier(df, dc)
    #special_reg(df, dc)
    #train_model_regression(df, dc)


if __name__ == '__main__':
    main()
