import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the extracted columns globally
extracted_cols = ['概念理解', '程序執行', '問題解決',
                  'absolution', 'trigonometric', 'log', 'exponential',
                  '運算程度', '閱讀理解']


def process_quiz_data(quiz_data_path, properties_data_path):
    # Step 1: Load the quiz response data
    quiz_data = pd.read_csv(quiz_data_path, encoding='Big5')
    quiz_data.fillna(0, inplace=True)

    # Clean column names for better readability
    quiz_data.columns = [
        f"col_{i}" if isinstance(col, str) and any(ord(char) > 127 for char in col) else col
        for i, col in enumerate(quiz_data.columns)
    ]

    # Define sections based on column indices
    section1_cols = quiz_data.columns[2:12]  # Columns 3 to 12
    section2_cols = quiz_data.columns[12:20]  # Columns 13 to 20
    section3_cols = quiz_data.columns[20:25]  # Columns 21 to 25

    # Convert the values in the defined sections to numeric (non-convertible values become NaN)
    quiz_data[section1_cols] = quiz_data[section1_cols].apply(pd.to_numeric, errors='coerce')
    quiz_data[section2_cols] = quiz_data[section2_cols].apply(pd.to_numeric, errors='coerce')
    quiz_data[section3_cols] = quiz_data[section3_cols].apply(pd.to_numeric, errors='coerce')
    #quiz_data.dropna(inplace=True)

    # Calculate scores for each section
    quiz_data['section1_score'] = quiz_data[section1_cols].sum(axis=1) * 2
    quiz_data['section2_score'] = quiz_data[section2_cols].sum(axis=1) * 4
    quiz_data['section3_score'] = quiz_data[section3_cols].sum(axis=1)
    quiz_data['total_score'] = quiz_data['section1_score'] + quiz_data['section2_score'] + quiz_data['section3_score']

    # Load the properties data
    properties_data = pd.read_csv(properties_data_path, encoding='Big5')
    # Extract the rows 1 to 23 for the specified columns
    extracted_columns = properties_data[extracted_cols].iloc[0:23]
    col_sums = extracted_columns.sum(axis=0)

    # Convert quiz response data columns 3 to 25 into a binary matrix
    binary_matrix = (quiz_data.iloc[:, 2:25] != 0).astype(int)
    binary_array = binary_matrix.to_numpy()
    extra_arrays = extracted_columns.to_numpy()

    # Perform matrix multiplication to get weighted scores for each extracted column
    results = np.matmul(binary_array, extra_arrays)

    # Assign the resulting scores to quiz_data using the same names as the extracted columns
    for idx, col_name in enumerate(extracted_cols):
        quiz_data[col_name] = results[:, idx]
    # Compute the correlation matrix between total_score and the new analysis columns
    analysis_columns = extracted_cols
    corr_matrix = quiz_data[['total_score'] + analysis_columns].corr()
    print("Correlation Matrix:")
    print(corr_matrix)

    # Calculate R² values (squared correlation coefficients) for each analysis column with total_score
    r_values = corr_matrix.loc['total_score', analysis_columns]
    r2_values = r_values ** 2
    comparison_df = pd.DataFrame({'Column Sum': col_sums, 'R2': r2_values})
    print("\nR2 Compared Table:")
    print(comparison_df)

    # Instead of summing the rows of the multiplication matrix, keep the raw matrix
    quiz_multiplication_matrix = np.column_stack((results, quiz_data['total_score'].to_numpy()))  # This is not summed across respondents
    print(quiz_multiplication_matrix)

    return quiz_data, corr_matrix, comparison_df, quiz_multiplication_matrix


# Specify the main folder and subfolders
main_folder = r'D:/博士班/中央大學計畫/微積分課程資料'
subfolders = ['110_A','110_B', '111_A','111_B', '112_A', '112_B']

respond_pattern = re.compile(r"Quiz(\d+)_Respond\.csv", re.IGNORECASE)
properties_pattern = re.compile(r"Quiz(\d+)_Properties\.csv", re.IGNORECASE)

# Create a summary output file to store all results
output_file = os.path.join(main_folder, "summary_results.csv")
with open(output_file, 'w', encoding='utf-8-sig') as out_f:
    out_f.write("Summary of Correlation Matrices, R2 Compared Tables, and Aggregated Multiplication Matrices\n")
    out_f.write("=" * 80 + "\n\n")

    # Dictionary to store subfolder-level accumulated matrices (raw matrices concatenated vertically)
    subfolder_multiplication_matrices = {}

    # Loop over each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(main_folder, subfolder)
        try:
            files = os.listdir(folder_path)
        except FileNotFoundError:
            print(f"Subfolder {folder_path} not found. Skipping...")
            continue

        # Initialize an accumulator list for this subfolder
        accumulated_mult_matrices = []

        # Dictionaries to store file paths by quiz number
        quiz_files = {}
        properties_files = {}

        for file in files:
            respond_match = respond_pattern.match(file)
            properties_match = properties_pattern.match(file)
            if respond_match:
                quiz_num = respond_match.group(1)
                quiz_files[quiz_num] = os.path.join(folder_path, file)
            if properties_match:
                quiz_num = properties_match.group(1)
                properties_files[quiz_num] = os.path.join(folder_path, file)

        common_quiz_nums = set(quiz_files.keys()) & set(properties_files.keys())
        if not common_quiz_nums:
            print(f"No matching file pairs found in subfolder {subfolder}.")
            continue

        # Initialize the accumulator for the subfolder to None
        accumulated_mult_matrix = None

        # Process each quiz pair
        for quiz_num in sorted(common_quiz_nums, key=int):
            print(f"Processing Quiz {quiz_num} in folder {subfolder}...")
            quiz_data, corr_matrix, comparison_df, quiz_mult_matrix = process_quiz_data(
                quiz_files[quiz_num], properties_files[quiz_num]
            )
            print(f"Quiz {quiz_num} processing complete.\n")

            # If the accumulator is empty, initialize it with the first quiz matrix
            if accumulated_mult_matrix is None:
                accumulated_mult_matrix = quiz_mult_matrix.copy()
            else:
                # Check if the shape matches
                if accumulated_mult_matrix.shape != quiz_mult_matrix.shape:
                    # Handle shape mismatch: you might decide to skip or pad/truncate
                    print(f"The original size is {accumulated_mult_matrix.shape}, the new size is {quiz_mult_matrix.shape}")
                    print(f"Shape mismatch for quiz {quiz_num} in subfolder {subfolder}.")
                    # For example, if you are sure they should be aligned, you could raise an error:
                    raise ValueError("Matrix shapes do not match.")
                else:
                    accumulated_mult_matrix = (accumulated_mult_matrix + quiz_mult_matrix).astype(np.int64)

            # Write each quiz's detailed results into the summary file
            out_f.write(f"Subfolder: {subfolder}, Quiz: {quiz_num}\n")
            out_f.write("-" * 80 + "\n")
            out_f.write("Correlation Matrix:\n")
            out_f.write(corr_matrix.to_csv())
            out_f.write("\nR2 Compared Table:\n")
            out_f.write(comparison_df.to_csv())
            out_f.write("\n" + "=" * 80 + "\n\n")

        # Save the processed (accumulated) matrix for this subfolder in the main folder
        subfolder_output_file = os.path.join(folder_path, "aggregated_multiplication_matrix.csv")
        if accumulated_mult_matrix.size > 0:
            agg_df = pd.DataFrame(accumulated_mult_matrix, columns=extracted_cols + ['total_score'])
            agg_df.to_csv(subfolder_output_file, index=False)
            out_f.write(
                f"Aggregated Multiplication Matrix for Subfolder {subfolder} saved to:\n{subfolder_output_file}\n")
            out_f.write(agg_df.to_csv(index=False))
            out_f.write("\n" + "=" * 80 + "\n\n")
            print(f"Aggregated Multiplication Matrix for Subfolder {subfolder}:\n{agg_df}\n")
            subfolder_multiplication_matrices[subfolder] = accumulated_mult_matrix
        else:
            print(f"No multiplication matrices were processed for subfolder {subfolder}.")
            subfolder_multiplication_matrices[subfolder] = None

# Optionally, if you want to see the final results in your environment:
print("Final Aggregated Multiplication Matrices per Subfolder:")
for subfolder, matrix in subfolder_multiplication_matrices.items():
    if matrix is not None:
        df = pd.DataFrame(matrix, columns=extracted_cols + ['total_score'])
        print(f"Subfolder {subfolder}:\n{df}\n")
    else:
        print(f"Subfolder {subfolder}: No data processed.\n")
