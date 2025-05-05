import os
import pandas as pd

# Define the directory path
directory = r"C:\Users\14037\Documents\InsuranceCalc\Work Product"

# Step 1: Look for CSVs and handle multiple files
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
if not csv_files:
    print("No CSV files found in the directory.")
    exit()
elif len(csv_files) > 1:
    print("Multiple CSV files found:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    choice = int(input("Enter the number of the CSV file to use: ")) - 1
    if choice < 0 or choice >= len(csv_files):
        print("Invalid choice. Exiting.")
        exit()
    selected_csv = csv_files[choice]
else:
    selected_csv = csv_files[0]

# Step 2: Rename the selected CSV to "1_Raw_Data.csv"
raw_data_path = os.path.join(directory, "1_Raw_Data.csv")
os.rename(os.path.join(directory, selected_csv), raw_data_path)
print(f"Renamed {selected_csv} to 1_Raw_Data.csv")

# Step 3: Load the raw CSV, treating "None" as a value
df = pd.read_csv(raw_data_path, keep_default_na=False)

# Step 4: Calculate expected columns
numeric_columns = ['age', 'bmi', 'children', 'charges']
categorical_columns = {
    'gender': 2, 'smoker': 2, 'region': 4, 'medical_history': 4,
    'family_medical_history': 4, 'exercise_frequency': 4, 'occupation': 4, 'coverage_level': 3
}
expected_columns = len(numeric_columns) + sum(categorical_columns.values())
print(f"Expected number of columns after one-hot encoding: {expected_columns}")

# Step 5: One-hot encode directly
df_encoded = pd.get_dummies(df, columns=categorical_columns.keys(), drop_first=False, dtype=int)

# Step 6: Save the processed CSV
processed_data_path = os.path.join(directory, "2_Processed_Data.csv")
df_encoded.to_csv(processed_data_path, index=False)

# Step 7: Check actual columns and confirm
actual_columns = df_encoded.shape[1]
print(f"Actual number of columns in processed CSV: {actual_columns}")
print(f"Saved one-hot encoded data to Work Product\\2_Processed_Data.csv")
if actual_columns == expected_columns:
    print("Column count matches the expectation!")
else:
    print(f"Column count mismatch! Expected {expected_columns}, got {actual_columns}.")