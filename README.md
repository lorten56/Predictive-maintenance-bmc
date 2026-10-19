# Predictive Maintenance Digital Twin — enhanced Streamlit dashboard

This synthetic-data demonstration includes a model comparison (Random Forest, Logistic Regression and Gradient Boosting), chronological train/validation/test partitions with seven-day purges, component status and last corrective repair, MTBF/MTTR, Kaplan–Meier first-failure survival estimates and an interactive illustrative replacement-cost scenario.

## Deploy
Upload `app.py`, `requirements.txt`, and the `data/` folder contents to the **root of the same GitHub repository**. In Streamlit Community Cloud select the repository and `app.py`. Changes committed to GitHub redeploy the app.

## Methodology and limitations
* Predictions: 2021–2023 training (through Dec 24), 2024 validation (through Dec 24), 2025 held-out test. Select best model by validation PR-AUC; select threshold by validation F1. Do not optimize on the test period. PR-AUC is more useful than accuracy for rare failures.
* MTBF: logged machine operating hours divided by recorded failure events. MTTR: mean recorded corrective repair duration. Both depend on complete event histories and definitions.
* Survival: Kaplan–Meier estimate using first observed failure for each machine/component series; all others right-censored. Very small and heterogeneous cohorts make estimates unreliable for real-world decisions. Observed age is time since first record, **not actual component age**.
* Costs: user-entered scenario values, not dataset estimates. Keeping cost = estimated conditional failure probability × failure consequence + daily operating premium × horizon. Replacement cost = planned removal/replacement + expected replacement failure cost. Excludes discounting and repeat failures.
* This is not a live dashboard and should not trigger maintenance work orders.
