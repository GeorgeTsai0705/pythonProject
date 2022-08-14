#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np


class course_data(pd.DataFrame):
    def just_print(self):
        print("Ha~ Ha~ Ha~")

    def select_semseter(self, semester):
        # compare both semesters together
        if semester not in [1,2]:
            return None

        # choose the required semester
        temp_index = []
        sem = self[["semester"]].values.reshape(-1, )
        for i in range(len(sem)):
            if sem[i] != semester:
                temp_index.append(i)
        self.drop(temp_index, inplace=True)
        return None

    def select_freshmen(self, including_senior):
        # consider all students
        if including_senior:
            return None
        else:
            # choose the freshman's data
            temp_index = []
            year = self[["course year"]].values.reshape(-1, )[0]
            stu_id = self[["student number"]].values.reshape(-1, )
            for i in range(len(stu_id)):
                if stu_id[i][0:3] != f'{year}':
                    temp_index.append(i)
            self.drop(temp_index, inplace=True)
            return None


    def draw_department_diagram(self, including_senior, semseter):
        if not including_senior:
            pass
        else:
            pass


def main():
    # load data from csv file
    df = pd.read_csv('processed_data/108.csv', encoding='unicode_escape')
    ime = course_data(data= df)
    ime.just_print()
    ime.select_semseter(semester=1)
    ime.select_freshmen(False)
    print(ime[["student number"]])

if __name__ == '__main__':
    main()