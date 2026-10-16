# Power BI Dashboard (in progress)

**Data source:** `../outputs/customer_churn_scores.csv` (one row per customer, scores are out-of-fold so they were not produced on data the model trained on).

| Column | Meaning |
|---|---|
| `CustomerID` | Customer identifier |
| `churn_probability` | Model-predicted probability of churn (0 to 1) |
| `predicted_churn` | 1 if probability >= chosen threshold (0.38) |
| `risk_band` | Low / Medium / High |
| `actual_churned` | Observed churn (1/0) |
| `Contract`, `Internet_Service`, `Tenure_Months`, `Monthly_Charges` | Key attributes for slicing |
| `annual_revenue` | 12 x monthly charges |
| `stated_churn_reason` | Reason given by customers who churned (blank otherwise) |

## Planned pages
1. **Executive overview:** churn rate, customers, revenue at risk, churn by contract/tenure.
2. **Churn drivers:** churn by service, payment method, tenure band; top stated reasons.
3. **Retention targeting:** high-risk customer list with revenue, filterable by contract and tenure.

Save the `.pbix` file in this folder when done and add screenshots to the root README.
