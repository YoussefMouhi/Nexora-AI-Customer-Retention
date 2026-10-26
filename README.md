# Nexora AI — Customer Retention Intelligence

An **exploratory customer-retention decision-support application** for subscription businesses. Train on historical churn outcomes, evaluate predictive reliability, score *separate active customers*, and export an outreach-priority list ranked by estimated monthly revenue exposure.

## The business problem

Subscription businesses cannot contact every customer with the same urgency. This tool estimates which active customers are more likely to cancel and how much recurring revenue is exposed, so customer-success teams can **prioritize investigation and outreach**. It does **not** prove which customers can be saved or how much revenue an intervention will recover.

## Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use `examples/historical_customers.csv` to train, then `examples/active_customers.csv` to generate priorities. The sample data are **synthetic**, not actual customer records. No default data are loaded.

## Data contract

Historical CSV: `CustomerID`, `Churn` (`Yes`/`No`, `Exited`/`Stayed`, `1`/`0`), `MonthlyRevenue`, and at least two of these supported **pre-outcome** features: `TenureMonths`, `ContractType`, `SupportTickets`, `PaymentMethod`, `InternetService`, `MonthlyRevenue`.

Active CSV: `CustomerID`, `MonthlyRevenue`, and **all model features used during training**, but **no churn outcome required**. Customer IDs must be unique.

Extra fields are ignored. Identifiers, arbitrary free text, and post-cancellation fields are never model inputs. This conservative allowlist is intentional: generic AutoML cannot reliably infer when a feature became available.

## Method

- 75%/25% stratified train/test split (random seed 42).
- 3-fold stratified CV **within training data** comparing logistic regression and random forest on ROC AUC.
- Numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding **inside each pipeline**.
- Held-out test metrics: ROC AUC, recall, precision, F1, accuracy; confusion matrix.
- Permutation importance measured on held-out test data (exploratory, not causal).
- Churn risk on active customers; expected monthly revenue exposure = probability × monthly recurring revenue. Rankings are by this estimated exposure.

## Limits and responsible use

- Random splits are not a substitute for time-based validation when dates are available. Only include features available *before* the predicted cancellation window.
- The probability scores are not explicitly calibrated; risk bands (35%, 65%) are illustrative. Validate calibration and thresholds on representative data.
- Historical selection bias, class imbalance, population drift, privacy, and fairness can undermine results. No guaranteed accuracy or retention uplift.
- A high churn probability is **not** evidence that a discount or intervention will change the outcome. Establish campaign effectiveness with controlled tests.
- Do not use for consequential automated decisions about individuals.
- The supplied synthetic data are demonstrations, not evidence of business impact.

## Tests

```bash
python -m pytest -q
```

## Project history and attribution

This focused retention application follows an earlier general-purpose Nexora AI Streamlit prototype adapted from an open-source project. The Apache 2.0 `LICENSE` is retained from that project. The original upstream repository URL should be added here before public publication if applicable; do not present upstream code as entirely original. This edition's retention-focused workflow, explicit data schema, outreach ranking and associated tests are distinct project work.
