# Data

**File:** `Telco_customer_churn.xlsx` (7,043 rows x 33 columns, one row per customer)

**Source:** IBM Telco Customer Churn sample dataset (fictional telecom company in California), distributed via IBM Cognos Analytics sample data and Kaggle.

## Column groups
| Group | Columns |
|---|---|
| Identifier | `CustomerID` |
| Location (not used in model) | `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` |
| Demographics | `Gender` (not used), `Senior Citizen`, `Partner`, `Dependents` |
| Account | `Tenure Months`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges` |
| Services | `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` |
| Target | `Churn Value` (1 = churned, 0 = stayed); `Churn Label` is the same as Yes/No |
| **Leakage: never use as features** | `Churn Score`, `Churn Reason`, `CLTV` (derived from or recorded after the churn outcome) |

## Known data quirks
- `Total Charges` is blank for 11 customers with tenure 0 (treated as 0).
- Add-on columns use "No internet service" / "No phone service", which are recoded to "No".
- All customers are in California, so location carries no signal here.

## Processed data
`processed/telco_clean.csv` is created by `notebooks/01_data_exploration.ipynb` (section 10) and read by `notebooks/02_churn_modeling.ipynb`. It contains 7,043 rows: `CustomerID`, 21 cleaned/engineered features, the target `Churn_Value`, and `Churn_Reason` (descriptive only, never a feature). Leakage columns, location columns and gender are removed. Do not edit it by hand; re-run notebook 01 instead.
