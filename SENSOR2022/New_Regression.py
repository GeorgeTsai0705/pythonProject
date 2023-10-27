import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def load_data():
    """
    Load spectral data and cPassResult data.
    Returns:
        X_data: Feature data
        Y_result: Target data
    """
    spectral_data = pd.read_csv('SpectralData.csv', header=None)
    c_pass_result = pd.read_csv('cPassResult.csv', header=None)

    wave_length = spectral_data.iloc[1:, 0].tolist()
    X_data = np.transpose(spectral_data.iloc[1:, 1:])

    # Assuming sample_names might be used in future, keeping it here
    sample_names = c_pass_result.iloc[0, 1:]
    Y_result = c_pass_result.iloc[3, 1:].values.ravel()

    return X_data, Y_result


def plot_predictions_vs_actual(y_test, predictions, model_name):
    """
    Plot predicted values against actual values.
    """

    y_test = list(map(float, y_test))
    print(y_test, predictions)
    plt.scatter(x= list(map(float, y_test)), y= predictions, c='r', marker='o', s=20)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
    print([min(y_test), max(y_test)], [min(y_test), max(y_test)])
    plt.title(f"{model_name} - Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    [min(y_test), max(y_test)], [min(y_test), max(y_test)]
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid(True)
    plt.show()


def plot_mse_comparison(results):
    """
    Plot MSE comparison for each model.
    """
    names = list(results.keys())
    mses = [res["MSE"] for res in results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(names, mses, color=['blue', 'green', 'red'])
    plt.title("MSE Comparison Among Models")
    plt.ylabel("MSE")
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid(axis='y')
    plt.show()


# Load data and split into training and test sets
X_data, Y_result = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_result, test_size=0.2, random_state=42)

# Define parameters for the models
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

# Create models with the specified parameters
models = {name: model_class(**params) for name, (model_class, params) in zip(MODEL_PARAMS.keys(), [(SVR, MODEL_PARAMS['SVR']), (RandomForestRegressor, MODEL_PARAMS['RandomForestRegressor']), (MLPRegressor, MODEL_PARAMS['MLPRegressor'])])}

# Train each model, make predictions, and collect results
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {
        "Predicted": y_pred,
        "MSE": mse
    }
    print(f"{name}: MSE: {mse}")
    plot_predictions_vs_actual(y_test, y_pred, name)

# Plot MSE comparison
plot_mse_comparison(results)