import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_and_label_data(directory):
    data_list = []
    labels = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            # Read the txt file into a DataFrame, skipping the first row
            data = pd.read_csv(file_path, delim_whitespace=True, header=0, names=['Wavelength', 'Intensity'])

            # Extract intensity values as features (1D array)
            intensity_values = data['Intensity'].values

            # Label the data based on the first character of the filename
            # Assuming three classes 'A', 'B', and 'C'
            label = filename[0]

            # Append the intensity values and label to their respective lists
            data_list.append(intensity_values)
            labels.append(label)

    # Convert lists to DataFrame
    all_data = pd.DataFrame(data_list)
    all_labels = pd.Series(labels)

    return all_data, all_labels

# Define the directory containing the .txt files
directory = 'D:\博士班\Conference\MEMS2025\Spectrum Sample\Interpolated_Output'

# Call the function and get the labeled data
X, y = read_and_label_data(directory)

# Ensure that the labels are correctly handled as categories
y = pd.Categorical(y)

# Standardize the data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.97)  # Keep 97% of variance, or you can specify the number of components
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# 1. Random Forest Classifier with GridSearchCV
rf_param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 1, 5, 10],
    'min_samples_split': [2, 5, 8]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)
y_pred_rf = rf_grid.best_estimator_.predict(X_test)


# 2. Support Vector Machine (SVM) with GridSearchCV
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, n_jobs=-1, verbose=2)
svm_grid.fit(X_train, y_train)
y_pred_svm = svm_grid.best_estimator_.predict(X_test)



# 3. k-Nearest Neighbors (k-NN) with GridSearchCV
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, n_jobs=-1, verbose=2)
knn_grid.fit(X_train, y_train)
y_pred_knn = knn_grid.best_estimator_.predict(X_test)

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Best Parameters for Random Forest:", rf_grid.best_params_)

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("Best Parameters for SVM:", svm_grid.best_params_)

print("\nk-NN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nk-NN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("Best Parameters for k-NN:", knn_grid.best_params_)

# Optionally, save the PCA-transformed data to a CSV file
pca_output_file = 'labeled_spectra_data_pca.csv'
X_pca_df = pd.DataFrame(X_pca)
X_pca_df['Label'] = y  # Add the labels back to the data for saving
X_pca_df.to_csv(pca_output_file, index=False)
