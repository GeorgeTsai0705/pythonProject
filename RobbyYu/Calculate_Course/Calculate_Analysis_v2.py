import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_quiz_data(quiz_data_path, properties_data_path):
    # Step 1: Load the quiz response data
    quiz_data = pd.read_csv(quiz_data_path, encoding='Big5')

    # Delete rows that contain any empty values
    quiz_data.dropna(inplace=True)

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

    # Drop any rows that now contain NaN values after conversion
    quiz_data.dropna(inplace=True)

    # Calculate scores for each section
    quiz_data['section1_score'] = quiz_data[section1_cols].sum(axis=1) * 2
    quiz_data['section2_score'] = quiz_data[section2_cols].sum(axis=1) * 4
    quiz_data['section3_score'] = quiz_data[section3_cols].sum(axis=1)

    # Calculate the total score
    quiz_data['total_score'] = quiz_data['section1_score'] + quiz_data['section2_score'] + quiz_data['section3_score']

    # Load the properties data
    properties_data = pd.read_csv(properties_data_path, encoding='Big5')

    # Extract columns 12 to 14 (indices 11 to 13) rows 1 to 23
    extracted_cols = ['概念理解', '程序執行', '問題解決',
                      'absolution', 'trigonometric', 'log', 'exponential',
                      '運算程度', '閱讀理解']
    extracted_columns = properties_data[extracted_cols].iloc[0:23]

    # Calculate column sums for each extracted column from the properties file
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

    # Create a comparison table that shows the column sum and R² for each extracted column
    comparison_df = pd.DataFrame({'Column Sum': col_sums, 'R2': r2_values})
    print("\nR2 Compared Table:")
    print(comparison_df)

    return quiz_data, corr_matrix, comparison_df

# Specify the main folder containing the subfolders (e.g., '110', '111', '112')
main_folder = r'D:/博士班/中央大學計畫/微積分課程資料'
subfolders = ['110', '111', '112']

# Regular expressions to match file patterns
respond_pattern = re.compile(r"Quiz(\d+)_Respond\.csv", re.IGNORECASE)
properties_pattern = re.compile(r"Quiz(\d+)_Properties\.csv", re.IGNORECASE)

# Create a summary output file to store all results
output_file = os.path.join(main_folder, "summary_results.csv")
with open(output_file, 'w', encoding='utf-8-sig') as out_f:
    out_f.write("Summary of Correlation Matrices and R2 Compared Tables\n")
    out_f.write("=" * 80 + "\n\n")

    # Loop over each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(main_folder, subfolder)
        try:
            files = os.listdir(folder_path)
        except FileNotFoundError:
            print(f"Subfolder {folder_path} not found. Skipping...")
            continue

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

        # Process each quiz pair
        for quiz_num in sorted(common_quiz_nums, key=int):
            print(f"Processing Quiz {quiz_num} in folder {subfolder}...")
            quiz_data, corr_matrix, comparison_df = process_quiz_data(
                quiz_files[quiz_num], properties_files[quiz_num]
            )
            print(f"Quiz {quiz_num} processing complete.\n")

            # Write a header for this quiz's results
            out_f.write(f"Subfolder: {subfolder}, Quiz: {quiz_num}\n")
            out_f.write("-" * 80 + "\n")

            # Write the Correlation Matrix
            out_f.write("Correlation Matrix:\n")
            out_f.write(corr_matrix.to_csv())
            out_f.write("\n")

            # Write the R2 Compared Table
            out_f.write("R2 Compared Table:\n")
            out_f.write(comparison_df.to_csv())
            out_f.write("\n" + "=" * 80 + "\n\n")