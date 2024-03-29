import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# rate the different methods
method = {"s": 6, 'z': 5, 'c': 4, 'g': 3, '0': 0, 'x': 0.5, 'y': 0.5}


def load_data(filename):
    df = pd.read_csv(filename)

    # get the draw method data from different Problem
    draw_data = df[['ProbA_D', 'ProbB_D', 'ProbC_D', 'ProbD_D']].values

    draw_list = []
    # Count the number of times each method is used
    for ele in draw_data:
        # [ use 0 method , use 1 method, use 2 method, use 5 method]
        draw_list.append([(ele == 0).sum(), (ele == 1).sum(), (ele == 2).sum(), (ele == 3).sum()])

    solution_data = df[['ProbA_C', 'ProbB_C', 'ProbC_C', 'ProbD_C']].values

    solution_list = []
    for ele in solution_data:
        # change the solution code into number
        solution_list.append(sum([method[ele[0]], method[ele[1]], method[ele[2]], method[ele[3]]]))

    rank_data = df[['before_score_group', 'after_score_group']].values

    rank_list = []

    # if student's ability become worse maker "0", else maker "1" (not change or make progress
    for ele in rank_data:
        if ele[0] >= ele[1]:
            rank_list.append(1)
        else:
            rank_list.append(0)
    y = np.array([rank_list]).reshape(-1, 1)

    X1 = np.array(draw_list).reshape(-1, 4)
    X2 = np.array(solution_list).reshape(-1, 1)
    X = np.hstack([X1, X2])

    return X, y


def SVM_training(X, y):
    # normalize data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # cut the data into Train set and Test set
    # train_X, test_X, train_y, test_y = train_test_split(X_norm,y, random_state=111, train_size=0.8)

    # use SVM to predict student's performance
    SVM = LinearSVC(random_state=0, tol=1e-5)
    scores = cross_val_score(SVM, X_norm, y.ravel(), cv=5, scoring='accuracy')

    # SVM model predict result
    print("SVM accuracy:", np.mean(scores))


def KNN_training(X, y):

    # normalize data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    best_score = 0
    best_neighbors = 0

    for i in range(1,11):
        # use SVM to predict student's performance
        knn = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn, X_norm, y.ravel(), cv=5, scoring='accuracy')

        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_neighbors = i

    # knn model predict result
    print("knn accuracy:", best_score, " n_neighbors: ", best_neighbors)

def main():
    X, y = load_data("Data.csv")
    SVM_training(X, y)
    KNN_training(X, y)


if __name__ == '__main__':
    main()
