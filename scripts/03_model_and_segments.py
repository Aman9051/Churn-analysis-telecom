"""
03_model_and_segments.py
EDA, binary classification model for churn, explainability
(permutation/ELI5-style importance), and customer segmentation
(At Risk / Loyal / Dormant) on the real Kaggle Telco dataset.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.inspection import permutation_importance

sns.set_style("whitegrid")
OUT = "outputs"

df = pd.read_csv("data/telco_churn_clean.csv")

# ---------- EDA charts ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.countplot(data=df, x="contract", hue="churn", ax=axes[0], palette="Set2")
axes[0].set_title("Churn Count by Contract Type")
axes[0].tick_params(axis='x', rotation=15)

churn_rate_by_contract = df.groupby("contract")["churn_flag"].mean().sort_values(ascending=False)
churn_rate_by_contract.plot(kind="bar", ax=axes[1], color="#d9534f")
axes[1].set_title("Churn Rate by Contract Type")
axes[1].set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_contract.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x="churn", y="tenure_months", hue="churn", legend=False, palette="Set3", ax=ax)
ax.set_title("Tenure Distribution by Churn Status")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_tenure.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.kdeplot(data=df, x="monthly_charges", hue="churn", fill=True, alpha=0.4, ax=ax)
ax.set_title("Monthly Charges Distribution by Churn Status")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_monthly_charges.png", dpi=150)
plt.close()

# ---------- Preprocessing ----------
model_df = df.copy()
cat_cols = [
    "gender", "senior_citizen", "partner", "dependents", "phone_service",
    "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method",
]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    model_df[c] = le.fit_transform(model_df[c])
    encoders[c] = le

feature_cols = [c for c in model_df.columns if c not in ["customer_id", "churn", "churn_flag"]]
X = model_df[feature_cols]
y = model_df["churn_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- Models ----------
log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")
log_reg.fit(X_train_s, y_train)
y_pred_lr = log_reg.predict(X_test_s)
y_proba_lr = log_reg.predict_proba(X_test_s)[:, 1]

rf = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42,
                             class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

def eval_model(name, y_true, y_pred, y_proba=None):
    res = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        res["roc_auc"] = roc_auc_score(y_true, y_proba)
    return res

results_df = pd.DataFrame([
    eval_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
    eval_model("Random Forest", y_test, y_pred_rf, y_proba_rf),
])
results_df.to_csv(f"{OUT}/model_performance.csv", index=False)
print(results_df)

# Pick best model by ROC-AUC for confusion matrix / explainability
best_name = results_df.loc[results_df["roc_auc"].idxmax(), "model"]
print("Best model:", best_name)

cm = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

with open(f"{OUT}/classification_report.txt", "w") as f:
    f.write("Random Forest Classification Report\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\n\nLogistic Regression Classification Report\n")
    f.write(classification_report(y_test, y_pred_lr))

# ---------- Explainability: permutation importance (ELI5-style) ----------
perm = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
imp_df.to_csv(f"{OUT}/feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 6))
top_imp = imp_df.head(10).sort_values("importance_mean")
ax.barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"],
        color="#5b8bd9")
ax.set_title("Top 10 Drivers of Churn (Permutation Importance)")
ax.set_xlabel("Importance (drop in model accuracy)")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

# ---------- Customer Segmentation: At Risk / Loyal / Dormant ----------
full_proba = rf.predict_proba(X)[:, 1]
seg_df = df.copy()
seg_df["churn_probability"] = full_proba

def segment(row):
    if row["churn_probability"] >= 0.55:
        return "At Risk"
    elif row["tenure_months"] <= 3 and row["contract"] == "Month-to-month":
        return "Dormant"
    else:
        return "Loyal"

seg_df["segment"] = seg_df.apply(segment, axis=1)
seg_summary = seg_df.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_churn_prob=("churn_probability", "mean"),
    avg_monthly_charges=("monthly_charges", "mean"),
    avg_tenure=("tenure_months", "mean"),
).round(2).sort_values("avg_churn_prob", ascending=False)
seg_summary.to_csv(f"{OUT}/segment_summary.csv")
seg_df[["customer_id", "churn_probability", "segment"]].to_csv(
    f"{OUT}/customer_segments.csv", index=False
)
print("\nSegment Summary:\n", seg_summary)

fig, ax = plt.subplots(figsize=(6, 4.5))
seg_df["segment"].value_counts().plot(kind="bar", color=["#d9534f", "#f0ad4e", "#5cb85c"], ax=ax)
ax.set_title("Customer Segments")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{OUT}/segments.png", dpi=150)
plt.close()

print("\nAll model artifacts and charts saved to", OUT)
