import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== Load Dataset ====
DATA_PATH = "cardio_train.csv"
df = pd.read_csv(DATA_PATH, sep=';') if DATA_PATH.endswith(".csv") else pd.read_csv(DATA_PATH)

# ==== Dataset Cleaning and Feature Engineering ====
# Field Unification and Derivative Features
# Transfer age from days to years
df["age_years"] = (df["age"] / 365.25).round(2)
# Calculate BMI（kg/m^2）
df["height_m"] = df["height"] / 100.0
df["bmi"] = df["weight"] / (df["height_m"] ** 2)

# Outlier Filtering
# Systolic Blood Pressure & Diastolic Blood Pressure
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 180)]
df = df[df["ap_hi"] >= df["ap_lo"]]

# Height and Weight
df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]

# BMI
df = df[(df["bmi"] >= 10) & (df["bmi"] <= 60)]

# Target variables and characteristics
target = "cardio"
num_features = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]
cat_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

# Delete Columns
df = df.drop(columns=["age", "height_m"], errors="ignore")
# Remove Missing Parts
df = df.dropna(subset=num_features + cat_features + [target]).copy()
df[target] = df[target].astype(int)

# ==== Define feature groups ====
LIFESTYLE = ["smoke", "alco", "active"]
BIOMED = ["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "bmi", "height", "weight"]
COMBINED = LIFESTYLE + BIOMED + ["gender"]

FEATURE_SETS = {
    "lifestyle_only": LIFESTYLE,
    "biomedical_only": BIOMED,
    "combined": COMBINED
}

# ==== Divide the training/test set ====
X_full = df[num_features + cat_features]
y = df[target]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# ==== Model Definition ====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_params = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, solver="lbfgs"),
        {"clf__C": [0.1, 1.0, 10.0]}
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"clf__max_depth": [5, 10, 15]}
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        {"clf__max_depth": [None, 10, 20]}
    ),
    "SVM": (
        SVC(kernel="rbf", probability=True, random_state=42),
        {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale", 0.01]}
    ),
    "MLP": (
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    learning_rate="adaptive", max_iter=300, early_stopping=True,
                    random_state=42),
        {"clf__alpha": [1e-4, 1e-3]}
    ),
}


# ==== For each feature group and model, preprocess + train + evaluate ====
all_results = []
trained_store = {}


def get_scores(estimator, X):
    # Uniformly obtain the positive class scores
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    else:
        # Fallback: Use predictive labels instead
        return estimator.predict(X)


# Run 3 Feature Groups
for fs_name, fs_cols in FEATURE_SETS.items():
    # Filter columns based on feature groups
    X_train = X_train_full[fs_cols].copy()
    X_test  = X_test_full[fs_cols].copy()

    # Dynamically identify numerical/category columns
    num_cols = [c for c in fs_cols if c in num_features]
    cat_cols = [c for c in fs_cols if c in cat_features]

    # Preprocess: Numerical standardization + One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    for model_name, (estimator, param_grid) in models_and_params.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", estimator)
        ])

        gscv = GridSearchCV(
            pipe, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True
        )
        gscv.fit(X_train, y_train)

        best = gscv.best_estimator_
        y_pred = best.predict(X_test)
        y_score = get_scores(best, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        except Exception:
            auc = np.nan

        all_results.append({
            "FeatureSet": fs_name,
            "Model": model_name,
            "BestParams": gscv.best_params_,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(auc, 4) if not np.isnan(auc) else None
        })

        trained_store[(fs_name, model_name)] = {
            "estimator": best, "X_test": X_test, "y_test": y_test
        }
    
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves – {fs_name}")
    ax.legend(loc="lower right", fontsize=8)
    roc_path = os.path.join(PLOTS_DIR, f"ROC_{fs_name}.png")
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

# ==== Result ====
res_df = pd.DataFrame(all_results).sort_values(
    by=["F1", "Recall", "Accuracy"], ascending=False
).reset_index(drop=True)

csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# Take the combination with the highest F1 as the best model
best_row = res_df.iloc[0]
best_key = (best_row["FeatureSet"], best_row["Model"])
best_pack = trained_store[best_key]
best_est = best_pack["estimator"]
X_test_best = best_pack["X_test"]
y_test_best = best_pack["y_test"]

# Save confusion matrix diagram
fig_cm, ax_cm = plt.subplots(figsize=(5, 4), dpi=150)
ConfusionMatrixDisplay.from_estimator(best_est, X_test_best, y_test_best, ax=ax_cm)
ax_cm.set_title(f"Confusion Matrix – {best_row['FeatureSet']} × {best_row['Model']}")
cm_path = os.path.join(PLOTS_DIR, f"CM_{best_row['FeatureSet']}_{best_row['Model']}.png")
fig_cm.tight_layout()
fig_cm.savefig(cm_path)
plt.close(fig_cm)

print("\nSaved files:")
print(f"  • Results CSV: {csv_path}")
print(f"  • ROC figures: {PLOTS_DIR}/ROC_*.png")
print(f"  • Confusion Matrix: {cm_path}")