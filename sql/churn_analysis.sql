-- =====================================================================================================
-- Customer churn analysis in SQL
--
-- Table:  customer_predictions  (one row per customer; output/customer_churn_predictions.csv)
-- Run:    python -m src.run_sql          (loads the CSV into SQLite, runs every query, writes output/sql_results.md)
-- Tests:  python -m pytest tests/test_sql.py   (each query is checked against an independent pandas calculation)
--
-- Conventions
--   * ChurnFlag is 1 for customers who have left, 0 for active ones, so SUM(ChurnFlag) counts churners
--     and AVG(ChurnFlag) is the churn rate.
--   * Risk and revenue-exposure questions use ACTIVE customers only (CustomerStatus = 'Active'):
--     customers who already churned cannot be retained. Validation of the model uses ALL customers.
--   * ChurnProbability is an out-of-fold prediction, so it is safe to compare with the observed outcome.
--   * Portability: percentages are written as 100.0 * x / y. In SQLite and SQL Server, 100 * x / y with
--     integer inputs truncates (1/2 = 0), so the decimal literal is deliberate. Window functions need SQLite >= 3.25.
--     SUBSTR/INSTR (q15) are SQLite and MySQL spellings; SQL Server uses SUBSTRING/CHARINDEX.
-- =====================================================================================================


-- name: q00_sanity_checks
-- question: Is the table safe to analyse? (every count below should be 0)
-- reasoning: Cheap guards that catch a broken load or an inconsistent export before any number is trusted.
SELECT COUNT(*) - COUNT(DISTINCT CustomerID)                                              AS duplicate_ids,
       SUM(CASE WHEN ChurnProbability < 0 OR ChurnProbability > 1 THEN 1 ELSE 0 END)       AS invalid_probabilities,
       SUM(CASE WHEN ChurnProbability IS NULL OR MonthlyCharges IS NULL
                     OR RiskCategory IS NULL THEN 1 ELSE 0 END)                            AS rows_with_missing_core_fields,
       SUM(CASE WHEN (RiskCategory = 'High' AND ChurnProbability <= 0.70)
                  OR (RiskCategory = 'Low'  AND ChurnProbability >= 0.30) THEN 1 ELSE 0 END) AS tier_inconsistencies,
       SUM(CASE WHEN CustomerStatus = 'Churned' AND ExpectedMonthlyRevenueAtRisk <> 0
                THEN 1 ELSE 0 END)                                                        AS churned_customers_with_exposure
FROM customer_predictions;


-- name: q01_headline
-- question: How many customers do we have, and how many are churning?
-- reasoning: The headline numbers every other result is read against.
SELECT COUNT(*)                                        AS total_customers,
       SUM(ChurnFlag)                                  AS churned_customers,
       COUNT(*) - SUM(ChurnFlag)                       AS active_customers,
       ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 2)     AS churn_rate_pct,
       ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly_charges,
       ROUND(AVG(tenure), 1)                           AS avg_tenure_months
FROM customer_predictions;


-- name: q02_churn_by_contract
-- question: Which contract types hold the churn, and how concentrated is it?
-- reasoning: Rate alone hides size. A window over the grouped totals adds each group's share of customers and of all
--            churners, and its lift (group rate divided by the overall rate) in a single pass.
SELECT Contract,
       COUNT(*)                                                                    AS customers,
       SUM(ChurnFlag)                                                              AS churned,
       ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1)                                 AS churn_rate_pct,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                          AS share_of_customers_pct,
       ROUND(100.0 * SUM(ChurnFlag) / SUM(SUM(ChurnFlag)) OVER (), 1)              AS share_of_churners_pct,
       ROUND((1.0 * SUM(ChurnFlag) / COUNT(*))
             / (1.0 * SUM(SUM(ChurnFlag)) OVER () / SUM(COUNT(*)) OVER ()), 2)     AS lift_vs_overall
FROM customer_predictions
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- name: q03_churn_by_payment_method
-- question: Does the way customers pay relate to churn?
-- reasoning: A CTE computes the overall rate once, so each method can be expressed as points above or below it.
WITH overall AS (
    SELECT 1.0 * SUM(ChurnFlag) / COUNT(*) AS rate FROM customer_predictions
)
SELECT PaymentMethod,
       COUNT(*)                                                                        AS customers,
       ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1)                                     AS churn_rate_pct,
       ROUND(100.0 * SUM(ChurnFlag) / COUNT(*) - 100.0 * (SELECT rate FROM overall), 1) AS points_vs_overall
FROM customer_predictions
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- name: q04_charges_by_churn_status
-- question: Do churned customers pay more or less than active ones, and how long did they stay?
-- reasoning: Compares the two populations on the two numeric fields business teams ask about first.
SELECT CustomerStatus,
       COUNT(*)                          AS customers,
       ROUND(AVG(MonthlyCharges), 2)     AS avg_monthly_charges,
       ROUND(SUM(MonthlyCharges), 2)     AS total_monthly_charges,
       ROUND(AVG(tenure), 1)             AS avg_tenure_months
FROM customer_predictions
GROUP BY CustomerStatus
ORDER BY CustomerStatus;


-- name: q05_churn_by_tenure_group
-- question: When in the customer lifecycle does churn happen?
-- reasoning: TenureGroupOrder is grouped and sorted on so the bands appear in lifecycle order, not alphabetically.
SELECT TenureGroup,
       COUNT(*)                                       AS customers,
       SUM(ChurnFlag)                                 AS churned,
       ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1)    AS churn_rate_pct
FROM customer_predictions
GROUP BY TenureGroup, TenureGroupOrder
ORDER BY TenureGroupOrder;


-- name: q06_contract_by_tenure_matrix
-- question: Is the tenure effect just the contract effect in disguise?
-- reasoning: Contract and tenure are related. Conditional aggregation (CASE inside SUM) pivots churn rate by tenure band
--            within each contract, so the two can be separated. A band with no customers would divide by zero, hence NULLIF.
SELECT Contract,
       ROUND(100.0 * SUM(CASE WHEN TenureGroup = '0-12'  THEN ChurnFlag END) / NULLIF(SUM(CASE WHEN TenureGroup = '0-12'  THEN 1 END), 0), 1) AS churn_pct_0_12,
       ROUND(100.0 * SUM(CASE WHEN TenureGroup = '13-24' THEN ChurnFlag END) / NULLIF(SUM(CASE WHEN TenureGroup = '13-24' THEN 1 END), 0), 1) AS churn_pct_13_24,
       ROUND(100.0 * SUM(CASE WHEN TenureGroup = '25-48' THEN ChurnFlag END) / NULLIF(SUM(CASE WHEN TenureGroup = '25-48' THEN 1 END), 0), 1) AS churn_pct_25_48,
       ROUND(100.0 * SUM(CASE WHEN TenureGroup = '49+'   THEN ChurnFlag END) / NULLIF(SUM(CASE WHEN TenureGroup = '49+'   THEN 1 END), 0), 1) AS churn_pct_49_plus
FROM customer_predictions
GROUP BY Contract
ORDER BY CASE Contract WHEN 'Month-to-month' THEN 1 WHEN 'One year' THEN 2 ELSE 3 END;


-- name: q07_fiber_vs_dsl_by_price_band
-- question: Is fiber's higher churn just a price effect?
-- reasoning: Fiber customers pay more, so comparing DSL and fiber overall confounds service type with price. Holding the
--            monthly-charge band fixed isolates the service type. Bands with no customers of a type show NULL.
SELECT ChargeBand,
       SUM(CASE WHEN InternetService = 'DSL' THEN 1 ELSE 0 END)                                                  AS dsl_customers,
       ROUND(100.0 * SUM(CASE WHEN InternetService = 'DSL' THEN ChurnFlag END)
             / NULLIF(SUM(CASE WHEN InternetService = 'DSL' THEN 1 END), 0), 1)                                 AS dsl_churn_pct,
       SUM(CASE WHEN InternetService = 'Fiber optic' THEN 1 ELSE 0 END)                                          AS fiber_customers,
       ROUND(100.0 * SUM(CASE WHEN InternetService = 'Fiber optic' THEN ChurnFlag END)
             / NULLIF(SUM(CASE WHEN InternetService = 'Fiber optic' THEN 1 END), 0), 1)                         AS fiber_churn_pct
FROM customer_predictions
WHERE InternetService <> 'No'
GROUP BY ChargeBand, ChargeBandOrder
ORDER BY ChargeBandOrder;


-- name: q08_service_count_and_churn
-- question: Do customers with more protective services (security, backup, device protection, tech support) churn less?
-- reasoning: Restricted to customers with internet, because customers without it cannot have these add-ons and would
--            otherwise pollute the zero group.
SELECT ProtectiveAddOnCount,
       COUNT(*)                                       AS customers,
       ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1)    AS churn_rate_pct
FROM customer_predictions
WHERE InternetService <> 'No'
GROUP BY ProtectiveAddOnCount
ORDER BY ProtectiveAddOnCount;


-- name: q09_risk_tier_validation
-- question: Do the risk tiers mean what they say?
-- reasoning: Compares the average predicted probability with the observed churn rate in each tier over ALL customers.
--            Because the probabilities are out-of-fold, agreement is evidence the tiers are honest.
SELECT RiskCategory,
       COUNT(*)                          AS customers,
       ROUND(AVG(ChurnProbability), 3)   AS mean_predicted,
       ROUND(AVG(ChurnFlag), 3)          AS observed_churn_rate
FROM customer_predictions
GROUP BY RiskCategory, RiskOrder
ORDER BY RiskOrder DESC;


-- name: q10_risk_decile_gains
-- question: How well does the model concentrate churners at the top of the ranking?
-- reasoning: NTILE cuts customers into ten equal groups from highest to lowest predicted risk. A second window then
--            accumulates the share of all churners captured, the classic gains table, computed entirely in SQL.
--            Ties are broken by CustomerID so the result is deterministic.
WITH ranked AS (
    SELECT CustomerID, ChurnFlag, ChurnProbability,
           NTILE(10) OVER (ORDER BY ChurnProbability DESC, CustomerID) AS risk_decile
    FROM customer_predictions
),
by_decile AS (
    SELECT risk_decile, COUNT(*) AS customers, SUM(ChurnFlag) AS churners
    FROM ranked
    GROUP BY risk_decile
)
SELECT risk_decile,
       customers,
       churners,
       ROUND(100.0 * churners / customers, 1)                                          AS churn_rate_pct,
       ROUND(100.0 * SUM(churners) OVER (ORDER BY risk_decile) / SUM(churners) OVER (), 1) AS cumulative_pct_of_churners_captured,
       ROUND((1.0 * churners / customers) / (1.0 * SUM(churners) OVER () / SUM(customers) OVER ()), 2) AS lift
FROM by_decile
ORDER BY risk_decile;


-- name: q11_active_risk_profile
-- question: Who are the active customers in each risk tier?
-- reasoning: Averaging a 1.0/0.0 CASE gives the share of a group with some attribute, so one grouped query profiles
--            each tier on tenure, price and the three attributes the model leans on most.
SELECT RiskCategory,
       COUNT(*)                                                                         AS active_customers,
       ROUND(AVG(tenure), 1)                                                            AS avg_tenure_months,
       ROUND(AVG(MonthlyCharges), 2)                                                    AS avg_monthly_charges,
       ROUND(100.0 * AVG(CASE WHEN Contract = 'Month-to-month'   THEN 1.0 ELSE 0.0 END), 1) AS pct_month_to_month,
       ROUND(100.0 * AVG(CASE WHEN InternetService = 'Fiber optic' THEN 1.0 ELSE 0.0 END), 1) AS pct_fiber,
       ROUND(100.0 * AVG(CASE WHEN PaymentMethod = 'Electronic check' THEN 1.0 ELSE 0.0 END), 1) AS pct_electronic_check
FROM customer_predictions
WHERE CustomerStatus = 'Active'
GROUP BY RiskCategory, RiskOrder
ORDER BY RiskOrder DESC;


-- name: q12_revenue_exposure_by_risk
-- question: How much monthly revenue is exposed to churn, and where does it sit?
-- reasoning: Active customers only. Reports both the full charges of each tier (the brief's "revenue at risk" is the
--            High row) and the probability-weighted exposure. The window turns each tier's charges into a share of the total.
SELECT RiskCategory,
       COUNT(*)                                                                      AS active_customers,
       ROUND(SUM(MonthlyCharges), 2)                                                 AS monthly_charges,
       ROUND(100.0 * SUM(MonthlyCharges) / SUM(SUM(MonthlyCharges)) OVER (), 1)      AS pct_of_active_monthly_charges,
       ROUND(SUM(ExpectedMonthlyRevenueAtRisk), 2)                                   AS expected_monthly_revenue_at_risk
FROM customer_predictions
WHERE CustomerStatus = 'Active'
GROUP BY RiskCategory, RiskOrder
ORDER BY RiskOrder DESC;


-- name: q13_retention_segments
-- question: Which risk-and-value segments deserve attention first?
-- reasoning: RANK over the grouped expected exposure orders the segments by how much revenue is at stake, which is not the
--            same as ordering by risk: a large Medium segment can outweigh a small High one.
SELECT RetentionSegment,
       COUNT(*)                                                       AS active_customers,
       ROUND(SUM(MonthlyCharges), 2)                                  AS monthly_charges,
       ROUND(AVG(ChurnProbability), 3)                                AS avg_probability,
       ROUND(SUM(ExpectedMonthlyRevenueAtRisk), 2)                    AS expected_revenue_at_risk,
       RANK() OVER (ORDER BY SUM(ExpectedMonthlyRevenueAtRisk) DESC)  AS exposure_rank
FROM customer_predictions
WHERE CustomerStatus = 'Active'
GROUP BY RetentionSegment
ORDER BY exposure_rank;


-- name: q14_priority_retention_list
-- question: If the retention team can only contact 20 customers this week, who are they?
-- reasoning: Ranks active customers by expected revenue at risk (probability x monthly charge) so both likelihood and value
--            count, and shows the drivers the model associates with each. ROW_NUMBER with a CustomerID tie-break keeps it deterministic.
SELECT ROW_NUMBER() OVER (ORDER BY ExpectedMonthlyRevenueAtRisk DESC, CustomerID) AS priority,
       CustomerID,
       RiskCategory,
       ValueSegment,
       ROUND(ChurnProbability, 3)            AS churn_probability,
       MonthlyCharges,
       ROUND(ExpectedMonthlyRevenueAtRisk, 2) AS expected_monthly_revenue_at_risk,
       TopRiskDriver1,
       TopRiskDriver2
FROM customer_predictions
WHERE CustomerStatus = 'Active'
ORDER BY priority
LIMIT 20;


-- name: q15_main_risk_drivers
-- question: Which feature is the biggest model-associated risk driver among active customers who are Medium or High risk?
-- reasoning: TopRiskDriver1 is stored as 'feature = value'. Cutting it at ' = ' groups by feature, so tenure = 1, 2 and 3 months
--            count as one driver (tenure) instead of fragmenting into many small rows. The CTE keeps the GROUP BY portable
--            (some databases do not allow grouping by a select-list alias). These are model associations, not causes.
WITH at_risk AS (
    SELECT SUBSTR(TopRiskDriver1, 1, INSTR(TopRiskDriver1, ' = ') - 1) AS driver_feature,
           ExpectedMonthlyRevenueAtRisk
    FROM customer_predictions
    WHERE CustomerStatus = 'Active' AND RiskCategory IN ('Medium', 'High')
)
SELECT driver_feature                                                 AS main_risk_driver,
       COUNT(*)                                                       AS active_at_risk_customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)             AS share_pct,
       ROUND(SUM(ExpectedMonthlyRevenueAtRisk), 2)                    AS expected_monthly_revenue_at_risk
FROM at_risk
GROUP BY driver_feature
ORDER BY active_at_risk_customers DESC, main_risk_driver
LIMIT 10;
