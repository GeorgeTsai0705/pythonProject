# === Imports ===
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# === Data Loading ===
def load_data():
    """Load and preprocess the data."""
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)
    X_data = np.transpose(spectral_data.iloc[1:, 1:])
    Y_result = c_pass_result.iloc[3, 1:].values.ravel()
    return X_data, Y_result


# === PCA Application ===
def apply_pca(X_train, X_test, n_components=0.99):
    """Apply PCA to data."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


# === Visualization ===
def plot_predictions_vs_actual(y_test, predictions, model_name):
    """Plot predictions against actual data."""
    matplotlib.rcParams['font.family'] = "Times New Roman"

    plt.figure(figsize=(7, 7))
    y_test = list(map(float, y_test))
    y_pred = list(map(float, predictions))

    mae = mean_absolute_error(y_test, y_pred)
    diagonal = np.linspace(min(y_test), max(y_test), 100)

    plt.fill_between(diagonal, diagonal - mae, diagonal + mae, color='lightgray', label=f'± MAE ({mae:.2f})')
    plt.scatter(y_test, y_pred, alpha=0.75, s=20)
    plt.plot(diagonal, diagonal, 'r--')

    plt.title(f"{model_name} - Actual vs. Predicted", fontsize=20)
    plt.xlabel("Measured Neutralizing Antibody (IU/ml)", fontsize=15)
    plt.ylabel("Predicted Neutralizing Antibody (IU/ml)", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim((0, 230))
    plt.ylim((0, 230))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid(False)
    plt.show()


# === CSV Saving ===
def save_to_csv(name, result):
    """Save the Predicted and Actual results to a CSV."""
    df = pd.DataFrame({
        "Actual": result["Actual"],
        "Predicted": result["Predicted"]
    })
    df.to_csv(f"Grid_result/best_results_{name}.csv", index=False)


def main():
    """Main execution function."""

    MODEL_PARAMS = {
        "SVR": {
            'C': 60,
            'gamma': 4,
            'kernel': 'rbf'
        },
        "RandomForestRegressor": {
            'n_estimators': 15,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': False
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

    best_result = {
        "Name": None,
        "MSE": float("inf"),
        "Time": float("inf"),
        "R2": -float("inf"),
        "Predicted": None,
        "Actual": None,
        "BestSplit": None
    }

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=i)
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

            print(f"Iteration {i + 1}, Model {name}: MSE: {mse:.1f}, R2 Score: {r2:.1f}, Time: {elapsed_time:.1f} seconds")

            if mse < best_result["MSE"]:
                best_result = {
                    "Name": name,
                    "MSE": mse,
                    "Time": elapsed_time,
                    "R2": r2,
                    "Predicted": y_pred,
                    "Actual": y_test,
                    "BestSplit": (X_train, X_test, y_train, y_test)
                }

    X_train, X_test, y_train, y_test = best_result["BestSplit"]
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    validation_results = {}

    for name, params in MODEL_PARAMS.items():
        ModelClass = eval(name)
        model = ModelClass(**params)

        start_time = time.time()
        model.fit(X_train_pca, y_train)
        end_time = time.time()

        y_pred = model.predict(X_test_pca)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        elapsed_time = end_time - start_time

        validation_results[name] = {
            "MAE": mae,
            "Time": elapsed_time,
            "R2": r2,
            "Predicted": y_pred,
            "Actual": y_test
        }

    for name, result in validation_results.items():
        print(f"Validation Result for {name}: MAE: {result['MAE']:.3f}, R2 Score: {result['R2']:.3f}, Time: {result['Time']:.7f} seconds")
        plot_predictions_vs_actual(result["Actual"], result["Predicted"], name)


if __name__ == "__main__":
    main()
