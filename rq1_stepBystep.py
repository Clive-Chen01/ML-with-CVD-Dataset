import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Successive Halving
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

PLOTS_DIR = "plots"; RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = "cardio_train.csv"
df = pd.read_csv(DATA_PATH, sep=';') if DATA_PATH.endswith(".csv") else pd.read_csv(DATA_PATH)

# --- 预处理（同你之前，略有健壮性增强） ---
df = df[(df["height"] > 0) & (df["weight"] > 0)]
df["age_years"] = (df["age"] / 365.25).round(2)
df["height_m"] = df["height"] / 100.0
eps = 1e-6
df["bmi"] = df["weight"] / (df["height_m"].clip(lower=eps) ** 2)

df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 180)]
df = df[df["ap_hi"] >= df["ap_lo"]]
df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]
df = df[(df["bmi"] >= 10) & (df["bmi"] <= 60)]

target = "cardio"
ALL_NUM = ["age_years","height","weight","ap_hi","ap_lo","bmi"]
ALL_CAT = ["gender","cholesterol","gluc","smoke","alco","active"]
df = df.drop(columns=["age","height_m"], errors="ignore").dropna(subset=ALL_NUM+ALL_CAT+[target]).copy()
df[target] = df[target].astype(int)

LIFESTYLE = ["smoke","alco","active"]
BIOMED = ["age_years","ap_hi","ap_lo","cholesterol","gluc","bmi","height","weight"]
FEATURE_SETS = {
    "lifestyle_only": LIFESTYLE,
    "biomedical_only": BIOMED,
    # 如需再比 combined，可加上：
    # "combined": LIFESTYLE + BIOMED + ["gender"]
}

X_full = df[ALL_NUM + ALL_CAT]; y = df[target]
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# 3 折分层交叉验证
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 缩小后的网格（足够区分表现，又能快很多）
models_and_params = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1200, solver="lbfgs"),
        {"clf__C": [0.1, 1.0]}
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"clf__max_depth": [8, 12]}
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
        {"clf__max_depth": [None, 12]}
    ),
    "SVM": (
        SVC(kernel="rbf", probability=False, random_state=42),
        {"clf__C": [0.5, 1.0], "clf__gamma": ["scale"]}
    ),
    "MLP": (
        MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                      learning_rate="adaptive", max_iter=200, early_stopping=True,
                      random_state=42),
        {"clf__alpha": [1e-4, 1e-3]}
    ),
}

def get_scores(estimator, X):
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    elif hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    else:
        return estimator.predict(X)

all_results, trained_store = [], {}

for fs_name, fs_cols in FEATURE_SETS.items():
    X_train = X_train_full[fs_cols].copy()
    X_test  = X_test_full[fs_cols].copy()
    num_cols = [c for c in fs_cols if c in ALL_NUM]
    cat_cols = [c for c in fs_cols if c in ALL_CAT]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    # Pipeline 缓存预处理
    for model_name, (estimator, param_grid) in models_and_params.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", estimator)],
                        memory="cache_dir")  # 开启缓存

        # 使用 Successive Halving
        hgs = HalvingGridSearchCV(
            pipe, param_grid=param_grid, scoring="f1", cv=cv,
            factor=2, resource="n_samples",   # 默认即可
            n_jobs=-1, refit=True, verbose=0
        )
        hgs.fit(X_train, y_train)

        best = hgs.best_estimator_
        y_pred = best.predict(X_test)
        y_score = get_scores(best, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            fig, ax = plt.subplots(figsize=(6,5), dpi=150)
            ax.plot(fpr, tpr); ax.plot([0,1],[0,1],"--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"ROC – {fs_name} × {model_name} (AUC={auc:.3f})")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, f"ROC_{fs_name}_{model_name}.png"))
            plt.close(fig)
        except Exception:
            auc = np.nan

        all_results.append({
            "FeatureSet": fs_name, "Model": model_name,
            "BestParams": hgs.best_params_,
            "Accuracy": round(acc,4), "Precision": round(prec,4),
            "Recall": round(rec,4), "F1": round(f1,4), "ROC_AUC": round(auc,4) if not np.isnan(auc) else None
        })
        trained_store[(fs_name, model_name)] = {"estimator": best, "X_test": X_test, "y_test": y_test}

# 结果保存
res_df = (pd.DataFrame(all_results)
          .sort_values(by=["F1","Recall","Accuracy"], ascending=False)
          .reset_index(drop=True))
csv_path = os.path.join(RESULTS_DIR, "model_comparison_fast.csv")
res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# 混淆矩阵（取 F1 最高）
best_row = res_df.iloc[0]
key = (best_row["FeatureSet"], best_row["Model"])
pack = trained_store[key]
fig_cm, ax_cm = plt.subplots(figsize=(5,4), dpi=150)
ConfusionMatrixDisplay.from_estimator(pack["estimator"], pack["X_test"], pack["y_test"], ax=ax_cm)
ax_cm.set_title(f"CM – {best_row['FeatureSet']} × {best_row['Model']}")
fig_cm.tight_layout()
cm_path = os.path.join(PLOTS_DIR, f"CM_{best_row['FeatureSet']}_{best_row['Model']}.png")
fig_cm.savefig(cm_path); plt.close(fig_cm)

print("Saved:")
print(f" • {csv_path}")
print(f" • {PLOTS_DIR}/ROC_*_*.png")
print(f" • {cm_path}")
