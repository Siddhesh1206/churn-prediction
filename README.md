# Telco Customer Churn Prediction & Retention Targeting

Predicting which telecom customers are likely to churn, explaining **why**, and deciding **who a retention team should contact first**, using Python and scikit-learn. A Power BI dashboard built on the model output is in progress (see `powerbi/`).

## Business problem
Acquiring a new customer costs far more than keeping an existing one. A retention team has a limited budget, so it needs to know which customers are most at risk and what is driving them to leave, rather than contacting everyone.

## Dataset
IBM Telco Customer Churn sample dataset (a fictional California telecom company): **7,043 customers**, 33 columns covering demographics, services, contract, billing and churn outcome. **26.5%** of customers churned. See [`data/README.md`](data/README.md).

## Approach
1. **Removed data leakage** (notebook 01, section 10). This version of the dataset contains `Churn Score` (another model's output), `Churn Reason` (only recorded for customers who already left), `CLTV` and the churn label columns. Using them would inflate performance without being usable in practice, so they are excluded from the features. `Churn Reason` is used only for descriptive analysis.
2. **Dropped non-predictive fields.** Location columns (constant or high-cardinality) and gender (no business justification, avoids bias risk).
3. **Cleaned and engineered features.** Fixed blank `Total Charges` for brand-new customers, recoded "No internet/phone service" to "No" for add-ons, and created `n_protection_services`, `n_streaming_services` and `is_new_customer`.
4. **Leakage-safe modelling.** Stratified 80/20 split; preprocessing lives inside a scikit-learn `Pipeline`, so cross-validation never sees test-fold information.
5. **Compared three models:** logistic regression (interpretable baseline), random forest, and gradient boosting tuned with `RandomizedSearchCV` (scored on PR-AUC).
6. **Explained the model** with permutation importance and logistic-regression odds ratios.
7. **Chose the contact threshold by business value**, using out-of-fold predictions on the training data only (see scenario below).

## Results (held-out test set, 1,409 customers)

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.860 | 0.849 | 0.640 |
| Random Forest | 0.858 | 0.854 | 0.664 |
| **Gradient Boosting (tuned)** | **0.864** | **0.858** | **0.671** |

PR-AUC from random guessing equals the churn rate (about 0.27), so the model is far above baseline.

![ROC and PR curves](outputs/roc_pr_curves.png)

### Lift
Customers ranked by predicted risk:

- The **top decile has a 76.6% churn rate**, 2.9x the average.
- The **top 20% of customers contains 51% of all churners.**
- The three lowest-risk deciles together contain only **2 of the 374 churners** (full table in `outputs/decile_analysis.csv`).

### What drives churn?
![Feature importance](outputs/feature_importance.png)

- **Contract type** is the strongest driver. Raw churn rates: month-to-month **42.7%**, one-year **11.3%**, two-year **2.8%**.
- **Tenure:** newer customers churn far more; in the logistic model, being in the first 12 months carries about 1.7x the odds of churning, holding other features constant.
- **Internet service:** fibre-optic customers churn at **41.9%** vs 19.0% for DSL, and have about 2x the odds of churning in the logistic model.
- **Dependents** and **no internet service** are strongly protective (odds ratios about 0.2 and 0.3).
- Among churned customers, the most common stated reasons relate to **competitor offers** (speed, data, better deal) and **support staff attitude**.

![Churn rate by feature](outputs/eda_churn_by_feature.png)

## Retention campaign scenario (assumption-based)
The threshold for contacting a customer is chosen to maximise net benefit under **illustrative assumptions** (editable in the notebook):

- Cost of contacting + discounting a customer: **$80**
- Share of contacted churners retained: **30%**
- Value of a saved customer: **12 x monthly charges**

At the chosen threshold (0.38), the model reaches **precision 0.59 and recall 0.71** on the test set. Under these assumptions, targeting by model gives roughly **$36K** net benefit on the test customers, vs **-$15K** if everyone were contacted. These figures are a scenario estimate based on stated assumptions, **not measured savings**.

![Net benefit vs threshold](outputs/net_benefit_curve.png)

## Limitations
- The data is a single snapshot with no time dimension, so the model cannot be validated on future periods.
- The dataset is a sample and may not reflect real-world customer behaviour.
- Campaign cost and save-rate figures are assumptions; real values should come from A/B tests.
- Probabilities were not calibrated.

## Project structure
```
Churn Prediction/
├── data/
│   ├── README.md                      # data source and dictionary notes
│   ├── Telco_customer_churn.xlsx      # raw data
│   └── processed/
│       └── telco_clean.csv            # cleaned data (created by notebook 01)
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA, leakage check, saves cleaned data
│   └── 02_churn_modeling.ipynb        # loads cleaned data: modelling, evaluation, export
├── outputs/                           # charts + CSVs from notebooks 01 and 02
│   └── customer_churn_scores.csv      # per-customer risk scores (Power BI source)
├── powerbi/                           # dashboard (in progress)
├── requirements.txt
└── README.md
```

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Run the notebooks **in order** in VS Code or Jupyter (Run All in each):
1. `notebooks/01_data_exploration.ipynb` explores the raw data and writes `data/processed/telco_clean.csv`.
2. `notebooks/02_churn_modeling.ipynb` loads that file, trains and evaluates the models, and writes charts and `customer_churn_scores.csv` to `outputs/`.

Pipeline: `raw xlsx -> 01 -> data/processed -> 02 -> outputs -> Power BI`

## Next steps
- Power BI dashboard: executive overview, churn drivers, and a customer risk list (see `powerbi/README.md`).
- Probability calibration and SHAP explanations.
- Test uplift modelling (who is *persuadable*), not just who is likely to churn.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter, Power BI (planned)

## Author
Siddhesh Kadam | [LinkedIn](https://linkedin.com/in/siddhesh12) | [GitHub](https://github.com/Siddhesh1206)
