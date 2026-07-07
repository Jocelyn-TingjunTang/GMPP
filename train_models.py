"""
GMPP Delay Risk Classification  —  LR / DT / RF Training Pipeline
==================================================================
Put this file in the SAME folder as GMPP_Project_Outcome.csv and run:

    python3 train_models.py

Outputs written to the same folder:
    model_comparison.csv          — Accuracy/Precision/Recall/F1/AUC-ROC,
                                    holdout + 5-fold CV, for all three models
    rf_feature_importance.csv     — Gini-based feature importance (for Power BI)
    best_hyperparameters.txt      — selected hyperparameters (for Methodology 4.4)
    GMPP_Project_Predictions.csv  — one row per project: features, OOF predicted
                                    delay probability, actual outcome, department
                                    reliability band  (for Power BI Page 6)

Requirements:  pandas, numpy, scikit-learn  (pip install scikit-learn pandas)
Python:        3.8+
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
RARE_DEPT_THRESHOLD = 5      # departments with fewer projects → grouped as "Other"
TEST_SIZE         = 0.20
N_CV_FOLDS        = 5

NUMERIC_FEATURES     = ["Planned_Duration_Days", "Log_WLC"]
CATEGORICAL_FEATURES = ["Department_Grouped", "Baseline_IPA_RAG"]

# Out-of-fold accuracy per department from the stratified fairness analysis
# (Section 7.1 / Table 5 in the report).  Used in GMPP_Project_Predictions.csv.
DEPT_OOF_ACCURACY = {
    "DFT":   0.862,
    "Other": 0.684,
    "DESNZ": 0.917,
    "DCMS":  0.636,
    "DEFRA": 0.700,
    "MOJ":   0.111,
    "DLUHC": 0.500,
    "DFE":   0.600,
    "DoH":   1.000,
}
RELIABILITY_THRESHOLD = 0.60   # departments below this get a warning flag


# ── Data loading ───────────────────────────────────────────────────────────────
def load_primary_set(csv_path="GMPP_Project_Outcome.csv"):
    """
    Load GMPP_Project_Outcome.csv and return the 'Schedule Slippage' subset
    with engineered features ready for the classifier pipeline.
    """
    df = pd.read_csv(csv_path)

    # Keep only projects labelled via genuine longitudinal schedule comparison
    primary = df[
        df["Label_Method"] == "Schedule Slippage (>=2 valid annual observations)"
    ].copy()

    # ── Feature engineering ──────────────────────────────────────────────────
    # Log-transform whole life cost to address right-skew
    primary["Log_WLC"] = np.log1p(primary["Baseline_WLC_GBPm"])

    # Group rare departments (< RARE_DEPT_THRESHOLD projects) into "Other"
    dept_counts   = primary["Department"].value_counts()
    keep_depts    = dept_counts[dept_counts >= RARE_DEPT_THRESHOLD].index
    primary["Department_Grouped"] = primary["Department"].where(
        primary["Department"].isin(keep_depts), "Other"
    )

    # Treat "Unknown" IPA_RAG as an explicit category, not missing data
    primary["Baseline_IPA_RAG"] = primary["Baseline_IPA_RAG"].fillna("Unknown")

    return primary


# ── Preprocessor factory ───────────────────────────────────────────────────────
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    Numeric:     median imputation  [+ StandardScaler for LR only]
    Categorical: OneHotEncoder (handle_unknown='ignore')
    Everything is fitted inside the pipeline so no leakage occurs.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps),                                  NUMERIC_FEATURES),
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])


# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate_on_holdout(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred),               4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "AUC_ROC":   round(roc_auc_score(y_test, y_proba),                4),
    }


# ── Model specifications ───────────────────────────────────────────────────────
def get_model_specs() -> dict:
    """
    Returns the three classifier specs.
    scale=True  → StandardScaler applied to numeric features (Logistic Regression).
    scale=False → tree-based models are scale-invariant.
    """
    return {
        "Logistic Regression": {
            "scale": True,
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "param_grid": {
                "clf__C":            [0.01, 0.1, 1, 10, 100],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "Decision Tree": {
            "scale": False,
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "clf__max_depth":         [2, 3, 4, 5, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__criterion":         ["gini", "entropy"],
                "clf__class_weight":      [None, "balanced"],
            },
        },
        "Random Forest": {
            "scale": False,
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "clf__n_estimators":  [100, 200, 300],
                "clf__max_depth":     [3, 5, None],
                "clf__max_features":  ["sqrt", "log2"],
                "clf__class_weight":  [None, "balanced"],
            },
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, "GMPP_Project_Outcome.csv")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    primary = load_primary_set(csv_path)

    X = primary[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = primary["Delayed"].astype(int)

    print("=" * 70)
    print(f"Primary analysis set  :  n = {len(primary)}")
    print(f"Class distribution    :  {y.value_counts().to_dict()}")
    print(f"Department groups     :  {sorted(primary['Department_Grouped'].unique())}")
    print("=" * 70)

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain : n={len(X_train)}  (Delayed={y_train.sum()})")
    print(f"Test  : n={len(X_test)}   (Delayed={y_test.sum()})\n")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── 3. Train all three models with grid search ────────────────────────────
    results        = []
    best_params    = {}
    fitted_models  = {}

    for name, spec in get_model_specs().items():
        print(f"Training {name} ...", end=" ", flush=True)

        pipe = Pipeline([
            ("prep", build_preprocessor(scale_numeric=spec["scale"])),
            ("clf",  spec["estimator"]),
        ])
        grid = GridSearchCV(
            pipe, spec["param_grid"],
            scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
        )
        grid.fit(X_train, y_train)

        best_model         = grid.best_estimator_
        fitted_models[name] = best_model
        best_params[name]  = grid.best_params_

        # Holdout metrics
        holdout = evaluate_on_holdout(best_model, X_test, y_test)

        # Cross-validated metrics on the TRAINING set
        cv_auc = cross_val_score(best_model, X_train, y_train,
                                 scoring="roc_auc", cv=cv)
        cv_f1  = cross_val_score(best_model, X_train, y_train,
                                 scoring="f1",      cv=cv)

        results.append({
            "Model":              name,
            "Holdout_Accuracy":   holdout["Accuracy"],
            "Holdout_Precision":  holdout["Precision"],
            "Holdout_Recall":     holdout["Recall"],
            "Holdout_F1":         holdout["F1"],
            "Holdout_AUC_ROC":    holdout["AUC_ROC"],
            "CV5_AUC_ROC_mean":   round(cv_auc.mean(), 4),
            "CV5_AUC_ROC_std":    round(cv_auc.std(),  4),
            "CV5_F1_mean":        round(cv_f1.mean(),  4),
            "CV5_F1_std":         round(cv_f1.std(),   4),
        })
        print(f"done  (holdout AUC-ROC={holdout['AUC_ROC']})")

    # ── 4. Print and save model comparison ───────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path   = os.path.join(script_dir, "model_comparison.csv")
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # ── 5. Save best hyperparameters ──────────────────────────────────────────
    hp_path = os.path.join(script_dir, "best_hyperparameters.txt")
    with open(hp_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters:\n{json.dumps(best_params, indent=2)}")

    # ── 6. RF Gini feature importance ─────────────────────────────────────────
    rf_model      = fitted_models["Random Forest"]
    feature_names = rf_model.named_steps["prep"].get_feature_names_out()
    importances   = rf_model.named_steps["clf"].feature_importances_

    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_path = os.path.join(script_dir, "rf_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)

    print(f"\nTop 10 RF Gini importances:")
    print(fi_df.head(10).to_string(index=False))

    # ── 7. Project-level predictions for Power BI Page 6 ─────────────────────
    # Use the full primary set (X, y) — not just the training split — so every
    # project gets an out-of-fold score from a fold that never saw it.
    print("\nComputing out-of-fold probabilities for all 108 projects ...", end=" ")

    rf_pipe_full = Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),
        ("clf",  RandomForestClassifier(
            n_estimators  = 300,
            max_depth     = 5,
            max_features  = "sqrt",
            class_weight  = None,
            random_state  = RANDOM_STATE,
        )),
    ])

    oof_proba = cross_val_predict(
        rf_pipe_full, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    print("done")

    # Assemble the predictions table
    pred_df = primary[[
        "Project_Name",
        "Department",
        "Department_Grouped",
        "Planned_Duration_Days",
        "Baseline_WLC_GBPm",
        "Baseline_IPA_RAG",
        "Baseline_Year",
        "Latest_Year",
        "Delayed",
    ]].copy()

    pred_df["OOF_Delay_Probability"] = oof_proba.round(3)
    pred_df["Actual_Outcome"]        = pred_df["Delayed"].map({1: "Delayed", 0: "On Time"})

    # Map department-level reliability from Table 5
    pred_df["Department_OOF_Accuracy"] = (
        pred_df["Department_Grouped"]
        .map(DEPT_OOF_ACCURACY)
        .round(3)
    )
    pred_df["Low_Reliability_Flag"] = (
        pred_df["Department_OOF_Accuracy"] < RELIABILITY_THRESHOLD
    )
    pred_df["Reliability_Label"] = pred_df["Low_Reliability_Flag"].map({
        True:  "⚠ Low reliability — mandatory human review",
        False: "✓ Within acceptable range",
    })

    # Risk tier based on OOF probability
    def risk_tier(p):
        if p >= 0.60:
            return "High"
        elif p >= 0.35:
            return "Medium"
        else:
            return "Low"

    pred_df["Risk_Tier"] = pred_df["OOF_Delay_Probability"].apply(risk_tier)

    # Final column order for Power BI
    pred_df = pred_df[[
        "Project_Name",
        "Department",
        "Department_Grouped",
        "Baseline_Year",
        "Latest_Year",
        "Planned_Duration_Days",
        "Baseline_WLC_GBPm",
        "Baseline_IPA_RAG",
        "OOF_Delay_Probability",
        "Risk_Tier",
        "Actual_Outcome",
        "Department_OOF_Accuracy",
        "Low_Reliability_Flag",
        "Reliability_Label",
    ]].sort_values("OOF_Delay_Probability", ascending=False).reset_index(drop=True)

    pred_path = os.path.join(script_dir, "GMPP_Project_Predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"\nGMPP_Project_Predictions.csv  →  {len(pred_df)} rows")
    print(pred_df[["Project_Name", "Department_Grouped",
                   "OOF_Delay_Probability", "Risk_Tier",
                   "Actual_Outcome", "Department_OOF_Accuracy",
                   "Low_Reliability_Flag"]].head(10).to_string(index=False))

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Files written:")
    for f in ["model_comparison.csv", "best_hyperparameters.txt",
              "rf_feature_importance.csv", "GMPP_Project_Predictions.csv"]:
        print(f"  {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
