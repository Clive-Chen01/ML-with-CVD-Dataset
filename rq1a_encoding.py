import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from data_preprocessing import load_processed_data, load_and_preprocess_data

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== Load Dataset ====
try:
    df, target, num_features, cat_features, feature_sets = load_processed_data("cardio_processed.csv")
except FileNotFoundError:
    print("Processed data not found, reprocessing...")
    df, target, num_features, cat_features, feature_sets = load_and_preprocess_data("cardio_train.csv")

FEATURE_SETS = feature_sets

X_full = df[num_features + cat_features]
y = df[target]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models_and_params = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, solver="lbfgs"),
        {"clf__C": [0.1, 1.0, 10.0]}
    ),
    "SVM": (
        SVC(kernel="rbf", probability=False, random_state=42), 
        {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale", 0.01]}
    ),
    "MLP": (
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    learning_rate="adaptive", max_iter=300, early_stopping=True,
                    random_state=42),
        {"clf__alpha": [1e-4, 1e-3]}
    ),
}


def split_cols(cols, num_features, cat_features):
    """Split columns into numerical and categorical"""
    num = [c for c in cols if c in num_features]
    cat = [c for c in cols if c in cat_features]
    return num, cat

def make_preprocessor(encoding, num_cols, cat_cols, scale_ordinal=False):
    if encoding == "OneHot":
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if scale_ordinal:
            return ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", Pipeline([("ord", cat), ("scaler", StandardScaler())]), cat_cols),
            ], remainder="drop")
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", cat, cat_cols),
    ], remainder="drop")

def init_scores():
    """Initialize score collection dictionary"""
    return {k: [] for k in ["acc","prec","rec","f1","auc","auprc"]}

def update_scores(scores, y_true, y_pred, s):
    scores["acc"].append(accuracy_score(y_true, y_pred))
    scores["prec"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
    scores["rec"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))
    scores["f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if s is not None:
        scores["auc"].append(roc_auc_score(y_true, s))
        scores["auprc"].append(average_precision_score(y_true, s))

def summarize(scores):
    def _m(xs): return float(np.mean(xs)) if xs else np.nan
    def _s(xs): return float(np.std(xs, ddof=1)) if len(xs) > 1 else np.nan
    return {k: _m(v) for k,v in scores.items()} | {f"{k}_std": _s(v) for k,v in scores.items()}

def pos_score(model, X):
    try:
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        elif hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return None
    except Exception:
        return None

def train_model(X, y, fs_name, model_name, estimator, grid, enc, num_cols, cat_cols, nested=True):
    print(f"  {enc} → {model_name} ({'Nested 3x3' if nested else 'CV=5'})")
    scores = init_scores()
    cv_outer = outer_cv if nested else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i,(tr,te) in enumerate(cv_outer.split(X,y),1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe = Pipeline([("preprocess", make_preprocessor(enc, num_cols, cat_cols)), ("clf", estimator)])
        if nested:
            pipe = GridSearchCV(pipe, param_grid=grid, scoring="f1_macro", cv=inner_cv, n_jobs=-1, refit=True)
            best = pipe.fit(Xtr, ytr).best_estimator_
        else:
            best = pipe.fit(Xtr, ytr)
        yhat = best.predict(Xte)
        s = pos_score(best, Xte)
        update_scores(scores, yte, yhat, s)

    stat = summarize(scores)
    print(f"    F1={stat['f1']:.4f}±{stat['f1_std']:.4f} | AUPRC={stat['auprc']:.4f}±{stat['auprc_std']:.4f}")
    return dict(FeatureSet=fs_name, Model=model_name, Encoding=enc,
                Accuracy=round(stat["acc"],4), Accuracy_std=round(stat["acc_std"],4),
                Precision=round(stat["prec"],4), Precision_std=round(stat["prec_std"],4),
                Recall=round(stat["rec"],4), Recall_std=round(stat["rec_std"],4),
                F1_macro=round(stat["f1"],4), F1_macro_std=round(stat["f1_std"],4),
                AUPRC=(round(stat["auprc"],4) if not np.isnan(stat["auprc"]) else None),
                AUPRC_std=(round(stat["auprc_std"],4) if not np.isnan(stat["auprc"]) else None),
                ROC_AUC=(round(stat["auc"],4) if not np.isnan(stat["auc"]) else None),
                ROC_AUC_std=(round(stat["auc_std"],4) if not np.isnan(stat["auc_std"]) else None))

print("=== Step1: encodingA/B test ===")
all_results = []

# Main training loop: combined uses nested CV, others use single-layer CV
plan = [("combined", True), ("lifestyle_only", False), ("biomedical_only", False)]
for fs_name, use_nested in plan:
    cols = FEATURE_SETS[fs_name]
    Xfs = X_train_full[cols]
    num_cols, cat_cols = split_cols(cols, num_features, cat_features)
    print(f"\nFeature set: {fs_name}  [{'Nested' if use_nested else 'CV=5'}]")
    for enc in ["OneHot","Ordinal"]:
        for mdl, (est, grid) in models_and_params.items():
            all_results.append(train_model(Xfs, y_train, fs_name, mdl, est, grid, enc, num_cols, cat_cols, nested=use_nested))

print("\n=== Results Analysis ===")

results_df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "rq1a_encoding_nested_cv.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Results saved to: {csv_path}")

rows = []  # summary dataframe
onehot_wins = ordinal_wins = ties = 0

for fs_name in FEATURE_SETS.keys():
    for model_name in models_and_params.keys():
        onehot = results_df[(results_df["FeatureSet"] == fs_name) &
                            (results_df["Model"] == model_name) &
                            (results_df["Encoding"] == "OneHot")]
        ordinal = results_df[(results_df["FeatureSet"] == fs_name) &
                             (results_df["Model"] == model_name) &
                             (results_df["Encoding"] == "Ordinal")]
        if len(onehot) == 0 or len(ordinal) == 0:
            continue

        onehot_f1     = float(onehot["F1_macro"].iloc[0])
        ordinal_f1    = float(ordinal["F1_macro"].iloc[0])
        onehot_f1_std = float(onehot["F1_macro_std"].iloc[0])
        ordinal_f1_std= float(ordinal["F1_macro_std"].iloc[0])

        mean_diff = onehot_f1 - ordinal_f1
        std_diff  = np.sqrt(onehot_f1_std**2 + ordinal_f1_std**2)

        if abs(mean_diff) <= std_diff:
            winner = "Tie"
            ties += 1
        elif mean_diff > 0:
            winner = "OneHot"
            onehot_wins += 1
        else:
            winner = "Ordinal"
            ordinal_wins += 1

        rows.append({
            "FeatureSet": fs_name,
            "Model": model_name,
            "Winner": winner,
            "Mean_Diff": round(mean_diff, 4)  
        })
print(rows)
summary_df = pd.DataFrame(rows)
total_combinations = len(summary_df)
onehot_not_worse = onehot_wins + ties
onehot_rate = onehot_not_worse / total_combinations if total_combinations else 0.0

print("\n=== Encoding Comparison Results ===")
print(f"Total combinations: {total_combinations}")
print(f"OneHot wins: {onehot_wins}, Ordinal wins: {ordinal_wins}, Ties: {ties}")
print(f"OneHot not worse rate: {onehot_rate:.1%}")

decision = "OneHotEncoder" if onehot_rate >= 0.8 else "OrdinalEncoder"

print(f"Recommended encoding: {decision}")

