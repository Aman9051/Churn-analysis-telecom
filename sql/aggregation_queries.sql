-- overall_churn_rate

SELECT
    churn,
    COUNT(*) AS num_customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 1) AS pct
FROM customers
GROUP BY churn;

-- avg_metrics_by_churn

SELECT
    churn,
    ROUND(AVG(tenure_months), 1)    AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2)    AS avg_total_charges
FROM customers
GROUP BY churn;

-- churn_by_contract

SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- churn_by_internet_service

SELECT
    internet_service,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- churn_by_payment_method

SELECT
    payment_method,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- churn_by_tech_support

SELECT
    tech_support,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tech_support
ORDER BY churn_rate_pct DESC;

-- tenure_buckets_churn

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

