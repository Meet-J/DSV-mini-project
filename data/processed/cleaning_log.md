# Data Cleaning Log

Raw: 7,043 rows x 21 columns. Clean: 7,043 rows x 21 columns. No rows were dropped.

## Transformations and decisions

### 1. Trim whitespace in text columns
- **What:** Removed leading/trailing spaces from 18 text columns.
- **Why:** Whitespace-only cells look like values but are really missing data; trailing spaces also create fake duplicate categories.
- **Rows/cells affected:** 11
- **Effect on analysis/modelling:** No category labels changed; blank TotalCharges cells became empty strings so they can be detected exactly in the next step.

### 2. TotalCharges: convert to numeric, set blanks to 0
- **What:** Converted TotalCharges from text to float. 11 blank cells were set to 0.0.
- **Why:** All blank rows have tenure = 0 (checked in code), i.e. customers who have not completed a billing cycle, so cumulative charges of 0 is the consistent value. Mean/median imputation would invent revenue that was never billed; dropping the rows would discard 0.16% of customers for no analytical gain. This is an interpretation of the pattern, not something the dataset states.
- **Rows/cells affected:** 11
- **Effect on analysis/modelling:** Column becomes numeric and usable in models and Power BI. All affected customers have Churn = No, so the target is unaffected; they are brand-new customers, so tenure-based charts will show a tenure-0 group of 11.

### 3. Kept 'No phone service' / 'No internet service' levels
- **What:** No change in the cleaned file.
- **Why:** These levels are valid and informative for EDA/Power BI. They duplicate PhoneService / InternetService, which adds collinearity for linear models.
- **Rows/cells affected:** 0
- **Effect on analysis/modelling:** Collapsed to 'No' later, inside feature engineering, where the reason can be documented next to the modelling code.

### 4. Kept SeniorCitizen as 0/1; Churn as Yes/No
- **What:** No change.
- **Why:** Already clean and unambiguous. Human-readable labels for Power BI and numeric encoding for models are produced downstream.
- **Rows/cells affected:** 0
- **Effect on analysis/modelling:** None.

## Validation checks passed on the cleaned data

- 21 columns as expected
- no missing values in any column
- no fully duplicated rows
- customerID is unique
- customerID format is ####-AAAAA for every row
- TotalCharges is numeric
- tenure within 0-72 months
- MonthlyCharges strictly positive
- TotalCharges non-negative
- SeniorCitizen only contains 0/1
- gender contains only expected categories
- Partner contains only expected categories
- Dependents contains only expected categories
- PhoneService contains only expected categories
- MultipleLines contains only expected categories
- InternetService contains only expected categories
- Contract contains only expected categories
- PaperlessBilling contains only expected categories
- PaymentMethod contains only expected categories
- Churn contains only expected categories
- MultipleLines is 'No phone service' exactly when PhoneService = No
- internet add-ons are 'No internet service' exactly when InternetService = No
