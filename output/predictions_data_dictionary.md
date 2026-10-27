# customer_churn_predictions.csv: data dictionary

One row per customer (CustomerID is the primary key).

| column | description |
|---|---|
| CustomerID | Unique customer identifier (primary key). |
| Churn | Actual outcome in the dataset snapshot: Yes = customer had left, No = still a customer. |
| ChurnFlag | 1 if Churn = Yes else 0. AVERAGE(ChurnFlag) is the churn rate. |
| CustomerStatus | 'Churned' or 'Active'. Risk metrics and revenue at risk should use Active customers only. |
| ChurnProbability | Predicted probability of churn from an out-of-fold model (the customer was not in that model's training data). |
| RiskCategory | Low (p < 0.30), Medium (0.30 <= p <= 0.70) or High (p > 0.70). Practical project thresholds; see output/risk_segmentation_report.md. |
| RiskOrder | 1/2/3 for Low/Medium/High. Use as the 'sort by column' for RiskCategory. |
| ValueSegment | 'High value' if MonthlyCharges is in the top quartile of all customers, else 'Standard value'. Revenue is the only value signal in the data. |
| RetentionSegment | Risk tier + value segment for Active customers (e.g. 'High risk + high value'); 'Already churned' otherwise. |
| MonthlyCharges | Current monthly charge (currency not stated in the dataset). Not a model input. |
| TotalCharges | Cumulative charges. 0 for the 11 customers with tenure 0 (see data/processed/cleaning_log.md). Not a model input. |
| ExpectedMonthlyRevenueAtRisk | ChurnProbability x MonthlyCharges for Active customers; 0 for Churned customers. Assumes calibrated probabilities. |
| tenure | Months with the company. |
| TenureGroup | 0-12, 13-24, 25-48 or 49+ months. |
| TenureGroupOrder | 1-4; sort order for TenureGroup. |
| Contract | Month-to-month, One year or Two year. |
| InternetService | DSL, Fiber optic or No. |
| PaymentMethod | Electronic check, Mailed check, Bank transfer (automatic) or Credit card (automatic). |
| PaperlessBilling | Yes or No. |
| AutoPay | Yes if PaymentMethod is an automatic bank transfer or credit card. |
| NumberOfServices | Count of services held: phone, internet, and each of the six add-ons set to Yes (0-8). |
| ProtectiveAddOnCount | How many of OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport are Yes (0-4). |
| ChargeBand | MonthlyCharges band: <=35, 35-65, 65-95, >95 (upper edge inclusive). |
| ChargeBandOrder | 1-4; sort order for ChargeBand. |
| gender | Male or Female. Not a model input. |
| SeniorCitizen | Yes or No. |
| Partner | Yes or No. |
| Dependents | Yes or No. |
| PhoneService | Yes or No. |
| MultipleLines | Yes, No, or 'No phone service'. |
| OnlineSecurity | Yes, No, or 'No internet service'. |
| OnlineBackup | Yes, No, or 'No internet service'. |
| DeviceProtection | Yes, No, or 'No internet service'. |
| TechSupport | Yes, No, or 'No internet service'. |
| StreamingTV | Yes, No, or 'No internet service'. |
| StreamingMovies | Yes, No, or 'No internet service'. |
| TopRiskDriver1 | Feature pushing this customer's predicted churn probability up the most, as 'feature = value' (tree SHAP). Empty if none pushes up. |
| TopRiskDriver2 | Second-largest upward driver. |
| TopRiskDriver3 | Third-largest upward driver. |
| TopProtectiveDriver1 | Feature pushing the probability down the most. These describe the model's associations, not causes. |
| DataSplit | 'Train' or 'Test': which side of the 80/20 split used for model selection the customer fell on. Probabilities here are out-of-fold for everyone. |
