import os
import cv2 as cv
from os import path

wafer_ID = "2224-108"

# Chip order setting
chip_order = {'D1': 1, 'D2': 2, 'C1': 3, 'C2': 4, 'C3': 5, 'C4': 6, 'C5': 7, 'C6': 8, 'C7': 9, 'C8': 10, 'B1': 11,
              'B2': 12, 'B3': 13, 'B4': 14, 'B5': 15, 'B6': 16, 'B7': 17, 'A2': 18, 'A3': 19}


def Make_map(wafer_ID):
    # list the dir name
    dirs = [d for d in os.listdir() if path.isdir(d)]

    # load the blank image of wafer map
    img = cv.imread("Wafer_map.png")

    cv.putText(img, wafer_ID, (95, 133), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

    chip_position = [(267, 336), (319, 336), (159, 444), (212, 444), (267, 444), (319, 444),
                     (376, 444), (427, 444), (476, 444), (159, 551), (212, 551), (267, 551),
                     (319, 551), (376, 551), (427, 551), (476, 551), (319, 660)]

    # search the target dir
    for dir_name in dirs:
        if wafer_ID in dir_name:

            with open(path.join(dir_name, f"{wafer_ID}.txt"), "r", encoding='utf-8') as f:
                Cont = 0
                for line in f.readlines():
                    s = line.split('\t')
                    Slit = int(s[4])
                    cv.putText(img, f"{Slit}", chip_position[Cont], cv.FONT_HERSHEY_TRIPLEX, 0.65, (0, 0, 0), 1,
                               cv.LINE_AA)
                    Cont += 1
            cv.imwrite(f"{dir_name}/{wafer_ID}.png", img)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def Make_DG_map(wafer_ID):
    # list the dir name
    dirs = [d for d in os.listdir() if path.isdir(d)]

    # load the blank image of wafer map
    img = cv.imread("New mask wafer map.JPG")

    cv.putText(img, wafer_ID, (1685, 241), cv.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 4, cv.LINE_AA)

    chip_position = [(1109,420), (1405,420), (527,1200), (800,1200), (1090,1200), (1379,1200),
                     (1659,1200), (1971,1200), (2231,1200), (527,1770), (800,1770),
                     (1090,1770), (1379,1770), (1659,1770), (1971,1770), (2231,1770),
                     (1109,2100), (1405,2100)]

    # search the target dir
    for dir_name in dirs:
        if wafer_ID in dir_name:

            with open(path.join(dir_name, f"{wafer_ID}.txt"), "r", encoding='utf-8') as f:
                Cont = 0
                for line in f.readlines():
                    s = line.split('\t')
                    DG = int(s[5])
                    cv.putText(img, f"{DG}", chip_position[Cont], cv.FONT_HERSHEY_TRIPLEX, 4, (0, 0, 0), 4,
                               cv.LINE_8)
                    Cont += 1
            cv.imwrite(f"{dir_name}/{wafer_ID}_DG.png", img)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    Make_map(wafer_ID)
    #Make_DG_map(wafer_ID)
    print("Finish!")


if __name__ == "__main__":
    main()
