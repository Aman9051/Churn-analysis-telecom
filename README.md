# 📊 Customer Churn Analysis — Telecom Industry

End-to-end data analytics project predicting customer churn for a telecom company and translating the model's findings into actionable retention strategies.

## 🎯 Objective

Predict churn and derive actionable strategies to retain users in a highly competitive telecom environment.

## 📁 Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — 7,043 customers, 21 attributes covering demographics, account/contract details, subscribed services, and churn label. Overall churn rate: **26.5%**.

## 🛠️ Tools & Tech Stack

| Category | Tools |
|---|---|
| Data wrangling | Python (Pandas, NumPy) |
| Aggregation | SQL (SQLite) |
| Modeling | Scikit-learn (Logistic Regression, Random Forest) |
| Explainability | Permutation Importance (ELI5-style) |
| Visualization | Matplotlib, Seaborn |
| Reporting | ReportLab (PDF), Jupyter Notebook |

## 📂 Repository Structure

```
churn-analysis-telecom/
├── data/
│   ├── Telco-Customer-Churn.csv      # raw Kaggle dataset
│   └── telco_churn_clean.csv         # cleaned, analysis-ready dataset
├── notebooks/
│   └── Telecom_Churn_Analysis.ipynb  # full end-to-end notebook
├── scripts/
│   ├── 01_data_cleaning.py
│   ├── 02_sql_aggregation.py
│   ├── 03_model_and_segments.py
│   ├── 04_build_notebook.py
│   └── 05_build_pdf_report.py
├── sql/
│   └── aggregation_queries.sql       # SQL deliverable
├── outputs/                          # charts, metrics, segment CSVs
├── reports/
│   └── Churn_Analysis_Report.pdf     # 2-page project report
├── requirements.txt
└── README.md
```

## ⚙️ How to Run

```bash
git clone https://github.com/<your-username>/churn-analysis-telecom.git
cd churn-analysis-telecom
pip install -r requirements.txt

python scripts/01_data_cleaning.py
python scripts/02_sql_aggregation.py
python scripts/03_model_and_segments.py
python scripts/05_build_pdf_report.py

# or open the full pipeline interactively:
jupyter notebook notebooks/Telecom_Churn_Analysis.ipynb
```

## 🔎 Pipeline

1. **Data Cleaning** — fixed `TotalCharges` (stored as text with 11 blank values for new customers), standardized column names.
2. **SQL Aggregation** — churn rate by contract, tenure bucket, internet service, payment method, and tech support, run via SQLite.
3. **EDA** — visualized churn patterns across contract type, tenure, and monthly charges.
4. **Modeling** — trained Logistic Regression (baseline) and Random Forest classifiers on a 75/25 split.
5. **Explainability** — permutation importance (ELI5-style) to rank churn drivers.
6. **Segmentation** — every customer scored with a churn probability and grouped into **At Risk / Loyal / Dormant**.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.747 | 0.515 | 0.801 | 0.627 | **0.844** |
| Random Forest | 0.776 | 0.564 | 0.690 | 0.620 | 0.839 |

**Top churn drivers:** `contract` type (dominant), `tenure`, `online security` enrollment.

**SQL-confirmed patterns:**
- Month-to-month contracts churn at **42.7%** vs. 11.3% (one-year) and 2.8% (two-year)
- Fiber-optic customers churn at **41.9%** vs. 19.0% (DSL)
- Electronic check payers churn at **45.3%** — the highest of any payment method
- Churn peaks in the first 12 months of tenure (**47.4%**)

**Customer segments:**

| Segment | Customers | Avg. Churn Probability | Avg. Monthly Charges |
|---|---|---|---|
| At Risk | 2,154 | 0.75 | $76.20 |
| Dormant | 263 | 0.35 | $33.69 |
| Loyal | 4,626 | 0.16 | $61.20 |

## ✅ Recommendations

1. Incentivize month-to-month customers to upgrade to annual/two-year contracts.
2. Proactively target the **At Risk** segment with retention offers.
3. Bundle online security / tech support into fiber-optic plans by default.
4. Encourage a shift away from electronic check payments toward auto-pay methods.
5. Strengthen onboarding in the first 90 days to reduce early-tenure churn.

## 📄 Full Report

See [`reports/Churn_Analysis_Report.pdf`](reports/Churn_Analysis_Report.pdf) for the complete 2-page write-up (Introduction, Abstract, Tools Used, Steps, Conclusion).

## 📜 License

This project uses the publicly available [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset for educational purposes.
