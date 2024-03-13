import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import time

def load_data():
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)
    X_data = np.transpose(spectral_data.iloc[1:, 1:])
    Y_result = c_pass_result.iloc[3, 1:].values.ravel()
    return X_data, Y_result

def apply_pca(X_train, X_test, n_components=5):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def plot_predictions_vs_actual(y_test, predictions, model_name):

    # Set the default font to "Times New Roman"
    matplotlib.rcParams['font.family'] = "Times New Roman"

    plt.figure(figsize=(7, 7))
    y_test = list(map(float, y_test))
    y_pred = list(map(float, predictions))
    plt.scatter(y_test, y_pred, alpha=0.75, s=20)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Diagonal line

    # Set the title with increased font size
    plt.title(f"{model_name} - Actual vs. Predicted", fontsize=20)

    # Set x and y labels with increased font size
    plt.xlabel("Measured Neutralizing Antibody (IU/ml)", fontsize=15)
    plt.ylabel("Predicted Neutralizing Antibody (IU/ml)", fontsize=15)

    # Set the tick parameters with increased font size
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.grid(False)
    plt.show()

def save_to_csv(name, result):
    """
    Save the Predicted and Actual results to a CSV.
    """
    df = pd.DataFrame({
        "Indices": result["Indices"],  # Add this line
        "Actual": result["Actual"],
        "Predicted": result["Predicted"]
    })
    df.to_csv(f"Grid_result/best_results_{name}.csv", index=False)


# Define parameters for the models
MODEL_PARAMS = {
    "SVR": {
        'C': 60,
        'gamma': 4,
        'kernel': 'rbf'
    },
    "RandomForestRegressor": {
        'n_estimators': 2,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'bootstrap': True
    },
    "MLPRegressor": {
        'hidden_layer_sizes': (3,),
        'activation': 'relu',
        'solver': 'lbfgs',
        'alpha': 0.05,
        'learning_rate': 'adaptive',
        'max_iter': 2500,
        'early_stopping': True
    }
}

X_data, Y_result = load_data()
best_results = {
    "SVR": {"MSE": float("inf"), "Time": float("inf"), "R2": -float("inf")},
    "RandomForestRegressor": {"MSE": float("inf"), "Time": float("inf"), "R2": -float("inf")},
    "MLPRegressor": {"MSE": float("inf"), "Time": float("inf"), "R2": -float("inf")}
}

for i in range(120):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_data, Y_result, np.arange(len(Y_result)), test_size=0.2, random_state=47)
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    for name, params in MODEL_PARAMS.items():
        ModelClass = eval(name)
        model = ModelClass(**params)

        start_time = time.time()
        model.fit(X_train_pca, y_train)
        end_time = time.time()

        y_pred = model.predict(X_test_pca)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        elapsed_time = end_time - start_time

        #print(f"Data Split {i + 1}, Model {name}: MSE: {mse}, R2 Score: {r2}, Time: {elapsed_time} seconds")

        if mse < best_results[name]["MSE"]:
            best_results[name] = {
                "MSE": mse,
                "Time": elapsed_time,
                "R2": r2,
                "Predicted": y_pred,
                "Actual": y_test,
                "Indices": idx_test  # Add this line to store the original indices
            }

for name, result in best_results.items():
    print(f"Best Results for {name}: MSE: {result['MSE']:.3f}, Time: {result['Time']:.6f} seconds, R2 Score: {result['R2']:.3f}")
    plot_predictions_vs_actual(result["Actual"], result["Predicted"], name)
    save_to_csv(name, result)
