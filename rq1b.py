import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

RANDOM_STATE = 42
N_SPLITS = 3 


# Load data
from data_preprocessing import load_processed_data, load_and_preprocess_data

try:
    df, target, num_features, cat_features, _ = load_processed_data("cardio_processed.csv")
except FileNotFoundError:
    print("Processed data not found, reprocessing...")
    df, target, num_features, cat_features, _ = load_and_preprocess_data("cardio_train.csv")

df.columns = df.columns.str.strip()

y = df["cardio"].astype(int)
num_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
X = df[num_cols + cat_cols].copy()


numeric_tf = Pipeline([("scaler", StandardScaler())])
categorical_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer([
    ("num", numeric_tf, num_cols),
    ("cat", categorical_tf, cat_cols),
])


models = {
    "LR": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}


strategies = ["none", "class_weight", "oversample", "undersample"]

def rebalance_train(X_tr, y_tr, how):
    if how == "none" or how == "class_weight":
        return X_tr, y_tr 

    pos_idx = y_tr[y_tr == 1].index
    neg_idx = y_tr[y_tr == 0].index
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    rng = np.random.default_rng(RANDOM_STATE)

    if how == "oversample":
        if n_pos < n_neg:
            extra = rng.choice(pos_idx, size=n_neg - n_pos, replace=True)
            X_bal = pd.concat([X_tr.loc[neg_idx], X_tr.loc[pos_idx], X_tr.loc[extra]])
            y_bal = pd.concat([y_tr.loc[neg_idx], y_tr.loc[pos_idx], y_tr.loc[extra]])
        else:
            extra = rng.choice(neg_idx, size=n_pos - n_neg, replace=True)
            X_bal = pd.concat([X_tr.loc[pos_idx], X_tr.loc[neg_idx], X_tr.loc[extra]])
            y_bal = pd.concat([y_tr.loc[pos_idx], y_tr.loc[neg_idx], y_tr.loc[extra]])

    elif how == "undersample":
        if n_pos < n_neg:
            keep = rng.choice(neg_idx, size=n_pos, replace=False)
            X_bal = pd.concat([X_tr.loc[keep], X_tr.loc[pos_idx]])
            y_bal = pd.concat([y_tr.loc[keep], y_tr.loc[pos_idx]])
        else:
            keep = rng.choice(pos_idx, size=n_neg, replace=False)
            X_bal = pd.concat([X_tr.loc[keep], X_tr.loc[neg_idx]])
            y_bal = pd.concat([y_tr.loc[keep], y_tr.loc[neg_idx]])

    order = rng.permutation(X_bal.index)
    return X_bal.loc[order], y_bal.loc[order]


def build_pipeline(model_name, strategy):
    if model_name == "LR":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        if strategy == "class_weight":
            clf.set_params(class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        if strategy == "class_weight":
            clf.set_params(class_weight="balanced")

    return Pipeline([("prep", preprocess), ("clf", clf)])

rows = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for m_name, _ in models.items():
    print(f"\n=== Model: {m_name} ===")
    for strat in strategies:
        print(f" -> Strategy: {strat}")
        fold = 0
        for tr, va in skf.split(X, y):
            fold += 1
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]

            X_bal, y_bal = rebalance_train(X_tr, y_tr, strat)

            pipe = build_pipeline(m_name, strat)
            pipe.fit(X_bal, y_bal)

            y_pred = pipe.predict(X_va)
  
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_prob = pipe.predict_proba(X_va)[:, 1]
            else:
                try:
                    y_prob = pipe.decision_function(X_va)
                except Exception:
                    y_prob = None

            rows.append({
                "model": m_name,
                "strategy": strat,
                "fold": fold,
                "accuracy": accuracy_score(y_va, y_pred),
                "precision": precision_score(y_va, y_pred, zero_division=0),
                "recall": recall_score(y_va, y_pred),
                "f1": f1_score(y_va, y_pred),
                "roc_auc": roc_auc_score(y_va, y_prob) if y_prob is not None else np.nan,
                "pr_auc": average_precision_score(y_va, y_prob) if y_prob is not None else np.nan,
            })

results = pd.DataFrame(rows)
results.head()

def agg(g):
    return pd.Series({
        "accuracy_mean": g["accuracy"].mean(), "accuracy_std": g["accuracy"].std(),
        "precision_mean": g["precision"].mean(), "precision_std": g["precision"].std(),
        "recall_mean": g["recall"].mean(),     "recall_std": g["recall"].std(),
        "f1_mean": g["f1"].mean(),             "f1_std": g["f1"].std(),
        "roc_auc_mean": g["roc_auc"].mean(),   "roc_auc_std": g["roc_auc"].std(),
        "pr_auc_mean": g["pr_auc"].mean(),     "pr_auc_std": g["pr_auc"].std(),
    })

summary = results.groupby(["model","strategy"]).apply(agg).reset_index()

base = summary[summary["strategy"]=="none"][["model","recall_mean","f1_mean"]]
base = base.rename(columns={"recall_mean":"recall_base","f1_mean":"f1_base"})
summary_delta = summary.merge(base, on="model", how="left")
summary_delta["delta_recall"] = summary_delta["recall_mean"] - summary_delta["recall_base"]
summary_delta["delta_f1"]     = summary_delta["f1_mean"]     - summary_delta["f1_base"]

summary_delta["rank"] = summary_delta.groupby("model")["recall_mean"].rank(ascending=False, method="first")
summary_delta["winner"] = np.where(summary_delta["rank"]==1, "✓", "")

summary_delta = summary_delta.sort_values(["model","recall_mean"], ascending=[True,False])
summary_delta.round(4)


results.to_csv("results/rq1b_results.csv", index=False)
summary_delta.to_csv("results/rq1b_summary.csv", index=False)



