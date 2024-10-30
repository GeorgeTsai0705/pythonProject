import cv2
import numpy as np
import os
from os import path
from sklearn import svm


def train_dataset():
    X = []
    Y = []

    # Prepare training data
    files = os.listdir("Datasets/1")
    for file in files:
        img = cv2.imread(os.path.join("Datasets/1", file), cv2.IMREAD_GRAYSCALE)
        X_data = np.reshape(img, -1)
        X.append(X_data)
        Y.append(1)

    files = os.listdir("Datasets/2")
    for file in files:
        img = cv2.imread(os.path.join("Datasets/2", file), cv2.IMREAD_GRAYSCALE)
        X_data = np.reshape(img, -1)
        X.append(X_data)
        Y.append(2)

    files = os.listdir("Datasets/3")
    for file in files:
        img = cv2.imread(os.path.join("Datasets/3", file), cv2.IMREAD_GRAYSCALE)
        X_data = np.reshape(img, -1)
        X.append(X_data)
        Y.append(3)

    files = os.listdir("Datasets/5")
    for file in files:
        img = cv2.imread(os.path.join("Datasets/5", file), cv2.IMREAD_GRAYSCALE)
        X_data = np.reshape(img, -1)
        X.append(X_data)
        Y.append(5)

    files = os.listdir("Datasets/123")
    for file in files:
        img = cv2.imread(os.path.join("Datasets/123", file), cv2.IMREAD_GRAYSCALE)
        X_data = np.reshape(img, -1)
        X.append(X_data)
        Y.append(123)

    clf = svm.SVC()
    clf.fit(X, Y)
    return clf


def image_process(contours, thresh, clf):
    Results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 35:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w > h:
            continue

        crop_img = thresh[y:y + h, x:x + w]
        crop_img2 = cv2.resize(crop_img, (12, 24), interpolation=cv2.INTER_NEAREST)
        Test_data = np.reshape(crop_img2, -1)
        Test = [Test_data]

        Result = clf.predict(Test)
        if Result != 123:
            Results.append((x, Result[0]))

        #cv2.imwrite(f"Datasets/{Cont}.jpg", crop_img2)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
        #cv2.imshow(f'{dir[11:22]}', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    Results.sort(key= lambda s:s[0])
    return Results[0][1]*10+Results[1][1]


def main():
    clf = train_dataset()

    # Reset the folder name in English for opencv
    file_path = "2022.03.10-2224-106(NEW)-OM\Coating"

    # options : Photolithography Dicing Coating

    # List the folder and filename
    dirs = [d for d in os.listdir(file_path) if path.isdir(f"{file_path}/{d}")]

    # Chip order setting
    chip_order = {'D1': 1, 'D2': 2, 'C1': 3, 'C2': 4, 'C3': 5, 'C4': 6, 'C5': 7, 'C6': 8, 'C7': 9, 'C8': 10, 'B1': 11,
                  'B2': 12, 'B3': 13, 'B4': 14, 'B5': 15, 'B6': 16, 'B7': 17, 'A2': 18, 'A3': 19}

    for dir in dirs:
        # delet the chinese characters in path
        if len(dir) >= 32:
            os.rename(path.join(file_path, dir), path.join(file_path, dir[:23]))

    # refresh the folder name
    dirs = [d for d in os.listdir(file_path) if path.isdir(f"{file_path}/{d}")]
    dirs.sort(key=lambda s: chip_order[s[20:22]])

    # open each image of slit to detect width of slit
    for elements in dirs:
        img = cv2.imread(f"{file_path}/{elements}/1.jpg")

        """
        cv2.imshow(f'{dir[11:22]}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        # Because the color of mark is red, we just use a red color filter to enhance the contours
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)

        # Search the contours of red color symbol
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Use image process to get the width of slit
        w_slit = image_process(contours, thresh, clf)

        # Save the w_slit into text
        txt_path = 'Slit_W.txt'
        with open(txt_path,'a') as f:
            f.write(f"{elements[20:22]}\t{w_slit}\n")


if __name__ == "__main__":
    main()
