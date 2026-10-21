# Home Credit Scorecard Model

Credit-risk scorecard for Home Credit Indonesia: predicts the probability that a loan applicant experiences
payment difficulties (`TARGET = 1`) and translates it into lending-policy recommendations.

## Data
The eight tables and the column dictionary were **provided for this project as files**. Their structure matches the
public *Home Credit Default Risk* dataset; its documentation
(https://www.kaggle.com/competitions/home-credit-default-risk/data) is used only as a reference for column meanings.

The data contain loan-level personal and financial information and are **not committed to Git**.
Place the files in `data/raw/` (compressed `.zip` / `.gz` copies are read directly):

```
application_train.csv  application_test.csv  bureau.csv  bureau_balance.csv  previous_application.csv
POS_CASH_balance.csv   credit_card_balance.csv  installments_payments.csv  HomeCredit_columns_description.csv
```

## Reproduce
Run the notebooks **in order**; each starts a fresh Python process so the full pipeline fits in about 4 GB of RAM.
```bash
pip install -r requirements.txt
export HC_DATA_DIR=data/raw          # optional; data/raw is the default location
# Steps 1-3: research, data understanding, validation split, EDA (about 7 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_research_data_eda.ipynb
# Modelling dataset + Step 4 preprocessing (about 10 min); needs the split created by notebook 01
jupyter nbconvert --to notebook --execute --inplace notebooks/02_dataset_preprocessing.ipynb
# Step 5 feature engineering review (about 27 min); needs the dataset built by notebook 02
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
# Steps 6-9 modelling and evaluation (about 1 hour on one CPU; faster on multi-core machines)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/04_modelling.ipynb
# Steps 10-11 business impact and explainability (about 10 min; needs the models saved by notebook 04)
jupyter nbconvert --to notebook --execute --inplace notebooks/05_business_impact.ipynb
```
The dataset can also be rebuilt on its own: `python -m src.build_dataset --data-dir data/raw --out data/processed`.

| Notebook | Content |
|---|---|
| `01_research_data_eda.ipynb` | Step 1 research, Step 2 schema inspection and data quality, validation split, Step 3 EDA |
| `02_dataset_preprocessing.ipynb` | Applicant-level dataset build, Step 4 preprocessing decisions, leakage test, preprocessing comparison |
| `03_feature_engineering.ipynb` | Step 5: decision-time checks, feature catalogue (IV, AUC, PSI), recency analysis, value of each history table |
| `05_business_impact.ipynb` | Steps 10-11: threshold and cost scenarios (illustrative), three-band policy, holdout confirmation, validated SHAP explanations, segments, fairness screen, score stability, recommendations and limitations |
| `04_modelling.ipynb` | Steps 6-9: validation recap, feature selection, imbalance handling, tuning, calibration, robustness, model selection, one-time holdout evaluation; saves models and predictions |

Set `HC_QUICK=1` to run notebook 04 as a 15% smoke test (results not valid, nothing saved).

## Modelling dataset (`data/processed/`)
| File | Content |
|---|---|
| `abt_train.parquet` | 307,511 labelled loans, one row per `SK_ID_CURR`: cleaned application fields, 103 engineered features, `TARGET`, `fold` (-1 = locked holdout, 0-4 = CV folds) |
| `abt_test.parquet` | 48,744 unlabelled loans for scoring, same features |
| `data_dictionary.csv` | Source table, formula and meaning of every engineered feature |
| `build_report.json` | Row counts, coverage and cleaning counts from the build |