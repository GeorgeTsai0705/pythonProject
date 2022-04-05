import sklearn
import pandas as pd

# rate the different methods
method = {"s":4, 'z':3, 'c':2, 'g':1, '0':-1, 'x':0.5, 'y':0.5}

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
        print(ele)

def main():
    load_data("Data.csv")


if __name__ == '__main__':
    main()
