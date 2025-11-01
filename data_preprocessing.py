# ============================================
# File: data_preprocessing.py
# Project: Breast Cancer Classification
# ============================================

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------------------------
# 1️⃣ Auto-detect dataset folder path
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "combined_cleaned_data.csv")

print(f"📂 Looking for CSV files in: {DATA_PATH}\n")

# -------------------------------------------------
# 2️⃣ Load all CSV files automatically
# -------------------------------------------------
datasets = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".csv"):
        path = os.path.join(DATA_PATH, file)
        try:
            df = pd.read_csv(path)
            datasets.append(df)
            print(f"✅ Loaded {file} → {df.shape}")
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

# -------------------------------------------------
# 3️⃣ Clean column names
# -------------------------------------------------
for i in range(len(datasets)):
    datasets[i].columns = (
        datasets[i]
        .columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

# -------------------------------------------------
# 4️⃣ Handle missing values
# -------------------------------------------------
for i in range(len(datasets)):
    df = datasets[i]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# -------------------------------------------------
# 5️⃣ Merge all datasets
# -------------------------------------------------
combined_data = pd.concat(datasets, axis=0, ignore_index=True)
print(f"\n📊 Combined dataset shape: {combined_data.shape}")

# -------------------------------------------------
# 6️⃣ Encode categorical variables
# -------------------------------------------------
label_enc = LabelEncoder()
for col in combined_data.select_dtypes(include=["object"]).columns:
    if col != "diagnosis":
        combined_data[col] = label_enc.fit_transform(combined_data[col].astype(str))

# -------------------------------------------------
# 7️⃣ Encode 'diagnosis' column separately
# -------------------------------------------------
if "diagnosis" in combined_data.columns:
    combined_data["diagnosis"] = combined_data["diagnosis"].replace(
        {"M": 1, "B": 0, "Malignant": 1, "Benign": 0}
    )

# -------------------------------------------------
# 8️⃣ Scale numerical features
# -------------------------------------------------
scaler = StandardScaler()
num_cols = combined_data.select_dtypes(include=["int64", "float64"]).columns
combined_data[num_cols] = scaler.fit_transform(combined_data[num_cols])

# -------------------------------------------------
# 9️⃣ Save cleaned dataset
# -------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
combined_data.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Cleaned dataset saved at: {OUTPUT_PATH}")

# -------------------------------------------------
# 🔟 Summary
# -------------------------------------------------
print("\n✅ Data preprocessing complete!")
print("Columns in final dataset:", len(combined_data.columns))
print("Rows in final dataset:", len(combined_data))
