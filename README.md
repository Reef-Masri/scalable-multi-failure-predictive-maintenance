# Scalable Multi-Failure Predictive Maintenance System with Confidence-Based Prioritization and Resilience to Unknown Failures

COE305 Machine Learning — Final Project

A machine learning pipeline that classifies multiple industrial machine failure types
from historical sensor data, with a single reusable scikit-learn preprocessing pipeline
shared across training, tuning, and explainability.

## Problem

Industrial predictive maintenance systems often detect only known failure patterns with
static models and assume sensor data is always reliable. This project builds a classifier
for multiple failure types that stays robust under class imbalance and is structured to
extend toward adaptive learning and unknown-failure handling.

## Dataset

- Source: [Kaggle — Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- 10,000 records, 10 raw columns
- Target: `Failure Type` (6 classes, ~97% "No Failure")
- Model inputs: 5 numeric sensor readings + `Type` (one-hot) = 8 features
- A copy is included at [`data/predictive_maintenance.csv`](data/predictive_maintenance.csv)

## Approach

1. **Preprocessing** — missing-value and duplicate checks (none found), IQR outlier removal
   (459 rows dropped → 9,541 remaining), `LabelEncoder` for the target, `OneHotEncoder` for
   `Type`, `StandardScaler` for numeric features — all wrapped in one `ColumnTransformer` /
   `Pipeline`.
2. **Feature ranking** — mutual information (Torque > Rotational speed > Tool wear >
   Air temp > Process temp).
3. **Modeling** — non-ensemble (Logistic Regression, Random Forest, SVM-RBF) and
   ensemble (LightGBM, XGBoost, CatBoost).
4. **Validation & tuning** — 5-fold Stratified K-Fold, `RandomizedSearchCV` with
   weighted-F1 scoring.
5. **Interpretability** — SHAP (TreeExplainer) on the tuned XGBoost model.
6. **Unsupervised cross-check** — K-Means (k=6).

## Results (weighted F1-Score, test set)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **LightGBM (tuned)** | **0.9900** | **0.9878** |
| CatBoost (tuned) | 0.9874 | 0.9840 |
| XGBoost (tuned) | 0.9869 | 0.9836 |
| Logistic Regression | 0.9801 | 0.9750 |
| Random Forest | 0.9806 | 0.9749 |
| SVM (RBF) | 0.9759 | 0.9662 |

**Final ranking:** LightGBM > CatBoost > XGBoost > Logistic Regression > Random Forest > SVM (RBF)

Ranking by weighted one-vs-rest ROC-AUC instead puts SVM and Logistic Regression on top —
an expected effect of the heavy class imbalance. F1 remains the primary metric here because
it matches the tuning objective and the goal of labeling the specific failure type.

K-Means (k=6): silhouette 0.1862, ARI 0.0005 — clusters group general operating states but
do not separate failure types without supervision.

## Files

| Path | Description |
|------|-------------|
| `predictive_maintenance.ipynb` | Full analysis notebook (rebuilt from a fresh Colab run) |
| `COE305_Final_Project_Report.docx` | Written project report |
| `data/predictive_maintenance.csv` | Dataset |
| `requirements.txt` | Python dependencies |

## Running

```bash
pip install -r requirements.txt
jupyter notebook predictive_maintenance.ipynb
```

The first notebook cell downloads the dataset via the Kaggle API. To skip that, point the
`pd.read_csv(...)` call at `data/predictive_maintenance.csv` instead.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, SHAP, Matplotlib, Seaborn.
