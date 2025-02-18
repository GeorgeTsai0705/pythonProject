import os
import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
from typing import Optional, Dict, Any

# --- Part 1: Chip Summary Creation ---

def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Quadratic function: a * x^2 + b * x + c.
    """
    return a * x**2 + b * x + c

def safe_load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def safe_load_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file into a pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_chips(base_folder: str) -> None:
    """
    Analyzes chip data from subfolders grouped by class (e.g., A_Class, B_Class, C_Class).
    For each chip folder (within each class folder) the function loads the required files,
    calculates features, and finally creates a summary DataFrame that is cleaned (rows with NaN removed)
    and saved as chip_summary.csv in the base_folder.
    """
    chip_summary = []

    # Custom peak wavelengths (modifiable by the user)
    custom_peak_wavelengths = [435.83, 546.07, 763.51, 811.53]

    # Iterate over each class folder (e.g., A_Class, B_Class, C_Class)
    for class_name in os.listdir(base_folder):
        class_folder = os.path.join(base_folder, class_name)
        if not os.path.isdir(class_folder):
            continue  # Skip non-directory items

        # Iterate over each chip folder within the current class folder
        for chip_id in os.listdir(class_folder):
            chip_folder = os.path.join(class_folder, chip_id)
            if not os.path.isdir(chip_folder):
                continue  # Skip files if present

            # Define file paths for required files in each chip folder
            ui_parameters_path = os.path.join(chip_folder, "UI_parameters.txt")
            spectrum_data_path = os.path.join(chip_folder, "spectrum_data.csv")
            fwhm_table_path = os.path.join(chip_folder, "fwhm_table.csv")

            # Load UI parameters (JSON expected)
            ui_parameters = safe_load_json(ui_parameters_path)
            if ui_parameters is None:
                continue

            # Load spectrum data and FWHM table
            spectrum_data = safe_load_csv(spectrum_data_path)
            if spectrum_data is None:
                continue

            fwhm_table = safe_load_csv(fwhm_table_path)
            if fwhm_table is None:
                continue

            # --- Wavelength to Pixel Coordinate Transformation ---
            if 'Wavelength (WL)' in ui_parameters:
                wl_coeffs = ui_parameters['Wavelength (WL)']
                if isinstance(wl_coeffs, list) and len(wl_coeffs) >= 3:
                    x_indices = np.array(spectrum_data.index)
                    a, b, c = wl_coeffs[:3]
                    spectrum_data['Wavelength_nm'] = a + b * x_indices + c * (x_indices**2)

            # --- Initialize Feature Variables ---
            feature1_intensity = None
            feature2_intensity = None
            feature3_intensity = None
            feature4_difference = None

            # --- Compute Features based on Spectrum Data ---
            if 'Wavelength_nm' in spectrum_data.columns:
                # Feature 1: Maximum intensity in the range [459, 530] nm
                filtered_spectrum1 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 459) &
                                                   (spectrum_data['Wavelength_nm'] <= 530)]
                if not filtered_spectrum1.empty:
                    feature1_intensity = filtered_spectrum1['Intensity'].max()

                # Feature 2: Maximum intensity in the range [600, 668] nm
                filtered_spectrum2 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 600) &
                                                   (spectrum_data['Wavelength_nm'] <= 668)]
                if not filtered_spectrum2.empty:
                    feature2_intensity = filtered_spectrum2['Intensity'].max()

                # Feature 3: Minimum intensity in the range [777, 792] nm
                filtered_spectrum3 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 777) &
                                                   (spectrum_data['Wavelength_nm'] <= 792)]
                if not filtered_spectrum3.empty:
                    feature3_intensity = filtered_spectrum3['Intensity'].min()

                # Feature 4: Quadratic fit to estimate FWHM differences at custom peaks
                f1_fitted_list = []
                f2_actual_list = []
                for peak_wavelength in custom_peak_wavelengths:
                    peak_data = spectrum_data[(spectrum_data['Wavelength_nm'] >= peak_wavelength - 3) &
                                              (spectrum_data['Wavelength_nm'] <= peak_wavelength + 3)]
                    if peak_data.empty:
                        continue

                    center_index = peak_data['Intensity'].idxmax()
                    start_index = max(0, center_index - 3)
                    end_index = center_index + 4
                    extended_peak_data = spectrum_data.iloc[start_index:end_index]
                    x_data = extended_peak_data['Wavelength_nm']
                    y_data = extended_peak_data['Intensity']

                    try:
                        params, _ = curve_fit(quadratic, x_data, y_data)
                        a_fit, b_fit, c_fit = params

                        fitted_peak_position = -b_fit / (2 * a_fit)
                        peak_intensity = quadratic(fitted_peak_position, a_fit, b_fit, c_fit)
                        half_peak_intensity = peak_intensity / 2

                        c_shifted = c_fit - half_peak_intensity
                        roots = np.roots([a_fit, b_fit, c_shifted])

                        if len(roots) == 2 and np.all(np.isreal(roots)):
                            roots = np.real(roots)
                            f1_fitted = abs(roots[1] - roots[0])
                            row_mask = fwhm_table['Wavelength (nm)'] == peak_wavelength
                            if row_mask.any():
                                f2_actual = fwhm_table.loc[row_mask, 'FWHM (nm)'].values[0]
                                f1_fitted_list.append(f1_fitted)
                                f2_actual_list.append(f2_actual)
                    except Exception as e:
                        print(f"Error fitting quadratic for chip {chip_id} in class {class_name}, "
                              f"peak {peak_wavelength}: {e}")

                if f1_fitted_list and f2_actual_list:
                    feature4_difference = np.mean(np.abs(np.array(f1_fitted_list) - np.array(f2_actual_list)))

            # --- Spectrum Summary Statistics ---
            spectrum_max = spectrum_data['Intensity'].max()
            if not spectrum_max:
                spectrum_max = np.nan

            feature1_ratio = feature1_intensity / spectrum_max if feature1_intensity is not None and spectrum_max != 0 else None
            feature2_ratio = feature2_intensity / spectrum_max if feature2_intensity is not None and spectrum_max != 0 else None
            feature3_ratio = feature3_intensity / spectrum_max if feature3_intensity is not None and spectrum_max != 0 else None

            # --- Feature 5: Derived Calculation ---
            feature5 = None
            gain = ui_parameters.get('Gain')
            exposure_time = ui_parameters.get('Exposure Time (ms)')
            if gain is not None and exposure_time is not None and spectrum_max and spectrum_max != 0:
                feature5 = ((gain / 32) * (exposure_time / 900)) / (spectrum_max / 4000)

            # --- Feature 6: Ratio in a Specific Wavelength Range ---
            feature6 = None
            if 'Wavelength_nm' in spectrum_data.columns and spectrum_max and spectrum_max != 0:
                filtered_spectrum6 = spectrum_data[(spectrum_data['Wavelength_nm'] >= 410) &
                                                   (spectrum_data['Wavelength_nm'] <= 550)]
                if not filtered_spectrum6.empty:
                    max_intensity_in_range = filtered_spectrum6['Intensity'].max()
                    feature6 = max_intensity_in_range / spectrum_max

            # --- Consolidate Spectrum Statistics ---
            spectrum_stats = {
                'spectrum_max': spectrum_max,
                'feature1_max_intensity': feature1_intensity,
                'feature2_max_intensity': feature2_intensity,
                'feature3_min_intensity': feature3_intensity,
                'feature1_ratio': feature1_ratio,
                'feature2_ratio': feature2_ratio,
                'feature3_ratio': feature3_ratio,
                'feature4_difference': feature4_difference,
                'feature5': feature5,
                'feature6': feature6
            }

            # --- Extract FWHM Statistics for Selected Wavelengths ---
            selected_wavelengths = [404.66, 435.83, 546.07, 578.01, 696.54, 763.51, 811.53, 841.81]
            filtered_fwhm = fwhm_table[fwhm_table['Wavelength (nm)'].isin(selected_wavelengths)]
            fwhm_stats = {
                'fwhm_min': filtered_fwhm['FWHM (nm)'].min() if not filtered_fwhm.empty else None,
                'fwhm_max': filtered_fwhm['FWHM (nm)'].max() if not filtered_fwhm.empty else None,
                'fwhm_mean': filtered_fwhm['FWHM (nm)'].mean() if not filtered_fwhm.empty else None,
                'fwhm_std': filtered_fwhm['FWHM (nm)'].std() if not filtered_fwhm.empty else None
            }

            # --- Extract UI Parameters for the Summary ---
            ui_stats = {
                'ROI': ui_parameters.get('ROI'),
                'Exposure Time (ms)': ui_parameters.get('Exposure Time (ms)'),
                'Gain': ui_parameters.get('Gain'),
                'Baseline Evaluation': ui_parameters.get('Baseline Evaluation'),
                'Wavelength (WL)': ui_parameters.get('Wavelength (WL)')
            }

            # --- Combine all information for the current chip ---
            chip_info = {
                'ChipID': chip_id,
                'Class': class_name,  # indicate chip's class
                **ui_stats,
                **spectrum_stats,
                **fwhm_stats
            }
            chip_summary.append(chip_info)

    # --- Create, Clean, and Save the Summary DataFrame ---
    summary_df = pd.DataFrame(chip_summary)
    # Replace empty strings with NaN and drop any rows with missing values.
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    summary_df.dropna(inplace=True)

    summary_output_path = os.path.join(base_folder, "chip_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Chip summary saved to {summary_output_path}")

# --- Part 2: ML Analysis on the Chip Summary ---

def find_best_thresholds_for_3classes(scores, true_labels):
    """
    Given the scores and true labels (A/B/C), performs a brute-force search for
    thresholds t1 < t2 that partition the scores into three groups (C, B, A)
    and returns the thresholds yielding the highest accuracy.
    """
    unique_scores = np.unique(scores)
    if len(unique_scores) < 3:
        return None, None, 0.0

    best_acc = 0.0
    best_t1, best_t2 = None, None

    for i in range(len(unique_scores) - 2):
        t1 = (unique_scores[i] + unique_scores[i + 1]) / 2.0
        for j in range(i + 1, len(unique_scores) - 1):
            t2 = (unique_scores[j] + unique_scores[j + 1]) / 2.0
            if t1 >= t2:
                continue

            pred_labels = []
            for s in scores:
                if s < t1:
                    pred_labels.append("C")
                elif s < t2:
                    pred_labels.append("B")
                else:
                    pred_labels.append("A")
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(true_labels, pred_labels)
            if acc > best_acc:
                best_acc = acc
                best_t1, best_t2 = t1, t2

    return best_t1, best_t2, best_acc

def ml_analysis(chip_summary_path: str) -> None:
    """
    Reads the chip_summary.csv file, normalizes selected features, trains a Random Forest
    classifier, determines optimized feature weights, searches for best thresholds to partition scores,
    and finally saves a CSV file with the computed scores and predictions.
    """
    # Required libraries for ML analysis
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import joblib

    # 1. Load CSV
    data = pd.read_csv(chip_summary_path)

    # 2. Select relevant columns (adjust if needed)
    columns_of_interest = [
        'fwhm_mean', 'feature1_ratio', 'feature2_ratio',
        'feature3_ratio', 'feature4_difference', 'feature5', 'feature6', 'fwhm_std', 'Baseline Evaluation'
    ]

    # 3. Subset of data
    data_subset = data[columns_of_interest]

    # 4. Record normalization parameters (min and max) for each column
    normalization_params = {
        column: {
            'max': data_subset[column].max(),
            'min': data_subset[column].min()
        }
        for column in columns_of_interest
    }

    # 5. Normalize each column to [0, 1] range. The logic "smaller is better" is preserved.
    normalized_data = (data_subset.max() - data_subset) / (data_subset.max() - data_subset.min())

    # (Optional) Print normalization parameters
    print("Normalization Parameters:")
    for column, params in normalization_params.items():
        print(f"{column}: max = {params['max']}, min = {params['min']}")

    # 6. Encode class labels (A/B/C) into integers.
    label_encoder = LabelEncoder()
    data['encoded_Class'] = label_encoder.fit_transform(data['Class'])

    # 7. Prepare features X and multi-class labels y.
    X = normalized_data
    y = data['encoded_Class']

    # 8. Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 9. Train a Random Forest classifier.
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # 10. Extract feature importances and convert to percentage weights.
    feature_importances = rf.feature_importances_
    importance_sum = np.sum(feature_importances)
    weights = {
        columns_of_interest[i]: feature_importances[i] / importance_sum
        for i in range(len(columns_of_interest))
    }

    print("Optimized Feature Weights:")
    for feature, weight in weights.items():
        print(f"{feature}: {weight:.4f}")

    # 11. Compute a weighted score for each record by summing (normalized value * feature weight)
    normalized_data['score'] = normalized_data.apply(
        lambda row: sum(row[col] * weights[col] for col in columns_of_interest),
        axis=1
    )
    data['score'] = normalized_data['score']

    scores = data['score'].values
    true_labels = data['Class'].values

    # 12. Find best thresholds for three classes based on the scores.
    t1, t2, best_acc = find_best_thresholds_for_3classes(scores, true_labels)
    print("Best T1 =", t1)
    print("Best T2 =", t2)
    print("Best Accuracy =", best_acc)

    # 13. Get final predicted labels based on thresholds.
    pred_labels = []
    for s in scores:
        if s < t1:
            pred_labels.append("C")
        elif s < t2:
            pred_labels.append("B")
        else:
            pred_labels.append("A")
    data['predicted_class_by_score'] = pred_labels

    # 14. Use Random Forest to predict on test set.
    y_pred = rf.predict(X_test)
    y_test_classes = label_encoder.inverse_transform(y_test)
    y_pred_classes = label_encoder.inverse_transform(y_pred)

    # 15. Compute multi-class ROC-AUC (using one-vs-rest).
    y_score = rf.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
    joblib.dump(rf, "random_forest_model.pkl")

    print("ROC AUC (multi-class, OVR):", roc_auc)
    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_classes, y_pred_classes))

    # 16. For completeness, predict on the full dataset.
    full_pred = rf.predict(normalized_data[columns_of_interest])
    data['predicted_Class'] = label_encoder.inverse_transform(full_pred)

    # 17. Save final data with scores and predictions.
    output_file_path = os.path.join(os.path.dirname(chip_summary_path), "ALL_with_scores_and_predictions.csv")
    data.to_csv(output_file_path, index=False)
    print(f"Data with scores and predictions saved to {output_file_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # Launch a file dialog to select the folder containing the class subfolders.
    Tk().withdraw()  # Hide the main Tk window
    folder_path = askdirectory(title="Select the folder containing A_Class, B_Class, C_Class")
    if folder_path:
        # First, generate and save the chip summary.
        analyze_chips(folder_path)
        # Then, perform the ML analysis on the generated chip_summary.csv.
        chip_summary_file = os.path.join(folder_path, "chip_summary.csv")
        ml_analysis(chip_summary_file)
    else:
        print("No folder selected.")
