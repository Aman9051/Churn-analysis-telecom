"""
02_sql_aggregation.py
Loads the cleaned Telco churn dataset into SQLite and runs SQL
aggregation queries (tenure, charges, services, contract) as
specified in the project mini-guide.
"""
import sqlite3
import pandas as pd

df = pd.read_csv("data/telco_churn_clean.csv")
conn = sqlite3.connect("data/telco_churn.db")
df.to_sql("customers", conn, if_exists="replace", index=False)

queries = {}

queries["overall_churn_rate"] = """
SELECT
    churn,
    COUNT(*) AS num_customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 1) AS pct
FROM customers
GROUP BY churn;
"""

queries["avg_metrics_by_churn"] = """
SELECT
    churn,
    ROUND(AVG(tenure_months), 1)    AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2)    AS avg_total_charges
FROM customers
GROUP BY churn;
"""

queries["churn_by_contract"] = """
SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;
"""

queries["churn_by_internet_service"] = """
SELECT
    internet_service,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;
"""

queries["churn_by_payment_method"] = """
SELECT
    payment_method,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;
"""

queries["churn_by_tech_support"] = """
SELECT
    tech_support,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tech_support
ORDER BY churn_rate_pct DESC;
"""

queries["tenure_buckets_churn"] = """
SELECT
    CASE
        WHEN tenure_months <= 12 THEN '0-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        WHEN tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_bucket,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY churn_rate_pct DESC;
"""

with open("outputs/sql_aggregation_results.txt", "w") as f:
    for name, q in queries.items():
        f.write(f"\n--- {name} ---\n")
        result = pd.read_sql_query(q, conn)
        f.write(result.to_string(index=False))
        f.write("\n")
        print(f"\n--- {name} ---")
        print(result.to_string(index=False))

with open("sql/aggregation_queries.sql", "w") as f:
    for name, q in queries.items():
        f.write(f"-- {name}\n{q}\n")

conn.close()
print("\nSaved SQL aggregation results.")
