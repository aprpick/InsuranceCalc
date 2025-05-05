import joblib
import pandas as pd

# **File paths**
model_path = "C:/Users/14037/Documents/InsuranceCalc/Work Product/3_Trained_Model.pkl"
data_path = "C:/Users/14037/Documents/InsuranceCalc/Work Product/2_Processed_Data.csv"
output_folder = "C:/Users/14037/Documents/InsuranceCalc/Model Training/Model Analysis"

# **Load trained model**
model = joblib.load(model_path)

# **Load feature names from processed CSV**
df = pd.read_csv(data_path)
feature_names = df.columns[:-1]  # Exclude target column

# **Check if model has weights**
if hasattr(model, "coef_"):
    weights = model.coef_

    # **Create DataFrame of raw feature weights**
    weights_df = pd.DataFrame({"Feature": feature_names, "Weight": weights})

    # **Sort from highest to lowest absolute weight**
    weights_df["Abs_Weight"] = weights_df["Weight"].abs()
    weights_df = weights_df.sort_values(by="Abs_Weight", ascending=False).drop(columns=["Abs_Weight"])

    # **Save CSV (Raw Feature Weights)**
    csv_path = f"{output_folder}/Feature_Weights.csv"
    weights_df.to_csv(csv_path, index=False)
    print(f"✅ Feature weights saved to: {csv_path}")

else:
    print("⚠️ Model does not have coefficients (weights).")
