"""
01_data_cleaning.py
Loads the raw Kaggle IBM Telco Customer Churn dataset, cleans it, and
saves an analysis-ready CSV.

Source dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import pandas as pd

RAW_PATH = "data/Telco-Customer-Churn.csv"
CLEAN_PATH = "data/telco_churn_clean.csv"

df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)

# TotalCharges is stored as text and has 11 blank values (new customers, tenure = 0)
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Standardize column names to snake_case for SQL / readability
df.columns = [
    "customer_id", "gender", "senior_citizen", "partner", "dependents",
    "tenure_months", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges", "churn",
]

df["churn_flag"] = (df["churn"] == "Yes").astype(int)
df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})

# Drop exact duplicates if any
before = len(df)
df = df.drop_duplicates(subset="customer_id")
print(f"Dropped {before - len(df)} duplicate customer_id rows")

df.to_csv(CLEAN_PATH, index=False)
print("Saved cleaned dataset:", df.shape)
print("Churn rate: {:.2%}".format(df["churn_flag"].mean()))
