
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from data_preprocessing import load_processed_data, load_and_preprocess_data

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== Dataset ====
try:
    df, target, num_features, cat_features, _ = load_processed_data("cardio_processed.csv")
except FileNotFoundError:
    print("Processed data not found, reprocessing...")
    df, target, num_features, cat_features, _ = load_and_preprocess_data("cardio_train.csv")

LIFESTYLE = ["smoke", "alco", "active"]
BIOMED = ["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "bmi", "height", "weight"]
COMBINED = LIFESTYLE + BIOMED + ["gender"]

FEATURE_SETS = {
    "lifestyle_only": LIFESTYLE,
    "biomedical_only": BIOMED,
    "combined": COMBINED
}

X_full = df[num_features + cat_features]
y = df[target].squeeze()
if isinstance(y, pd.Series):
    y = y.to_numpy() 

# ==== Hyperparameter Tuning + Full Dataset Evaluation ====
# Stage 1: Use subsampled data for hyperparameter tuning (faster)
# Stage 2: Use full dataset with best hyperparameters for final evaluation

X_full_original = X_full.copy()
y_original = y.copy()

# Stage 1
if len(y) > 10000: 
    N = 10000 
    test_size = 1 - (N / len(y)) 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    idx_sub, _ = next(sss.split(X_full, y))
    X_full = X_full.iloc[idx_sub]
    y = y[idx_sub]
    print(f"Stage 1: Subsampled to {len(y)} samples for hyperparameter tuning")
    print(f"Class distribution: {np.bincount(y)}")

# Split for hyperparameter tuning
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
single_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    'f1_macro': 'f1_macro',
    'accuracy': 'accuracy',
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}

SCALING_METHODS = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

FEATURE_SELECTION_STRATEGIES = {
    "none": None,

    "MI": {
        "selector": SelectKBest(mutual_info_classif, k="all"),
        "params": {"feature_selection__k": [3, 6, "all"]}  
    },

    "Chi2": {
        "selector": None,  
        "params": {"preprocess__cat__cat_sel__k": [3, 6, "all"]},
        "requires_non_negative": True
    },

    "ANOVA": {
        "selector": None,  
        "params": {"preprocess__num__num_sel__k": [3, 6, "all"]}
    },

    "L1": {
        "selector": None,
        "params": {"clf__C": [0.1, 1, 10]}  
    }
}


MODELS_CONFIG = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=2000, solver="saga", random_state=42),  
        "feature_selection": ["none", "MI", "Chi2", "ANOVA", "L1"],
        "base_params": {"clf__C": [0.1, 1.0, 10.0]}
    },
    "SVM": {
        "estimator": SVC(kernel="rbf", probability=False, random_state=42),  
        "feature_selection": ["none", "MI", "Chi2", "ANOVA"],
        "base_params": {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale", 0.01]}
    },
    "MLP": {
        "estimator": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                 learning_rate="adaptive", max_iter=300, early_stopping=True,
                                 random_state=42),
        "feature_selection": ["none", "MI", "Chi2", "ANOVA"],
        "base_params": {"clf__alpha": [1e-4, 1e-3]}
    }
}

def split_cols(cols, num_features, cat_features):
    num = [c for c in cols if c in num_features]
    cat = [c for c in cols if c in cat_features]
    return num, cat

def k_grid(max_k):
    base = [3, 6, 'all']
    return [k for k in base if k == 'all' or k <= max_k]

def make_preprocessor(scaling_method, num_cols, cat_cols, feature_selection_strategy=None):
    """OneHot encoding (decided by rq1a_encoding.py) and specified scaling"""
    scaler = SCALING_METHODS[scaling_method]
    
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    
    # For Chi2, apply feature selection only to categorical features
    if feature_selection_strategy == "Chi2":
        num_branch = Pipeline([("scaler", scaler)])  
        cat_branch = Pipeline([
            ("ohe", cat_encoder),
            ("cat_sel", SelectKBest(chi2, k=6))  
        ])
        return ColumnTransformer([
            ("num", num_branch, num_cols),
            ("cat", cat_branch, cat_cols),
        ], remainder="drop")
    
    # For ANOVA, apply feature selection only to numerical features (avoid sparse matrix issues)
    elif feature_selection_strategy == "ANOVA":
        num_branch = Pipeline([
            ("scaler", scaler),
            ("num_sel", SelectKBest(f_classif, k=6))  
        ])
        cat_branch = Pipeline([("ohe", cat_encoder)])  
        return ColumnTransformer([
            ("num", num_branch, num_cols),
            ("cat", cat_branch, cat_cols),
        ], remainder="drop")
    
    # For other feature selection methods, apply to all features after preprocessing
    return ColumnTransformer([
        ("num", scaler, num_cols),
        ("cat", cat_encoder, cat_cols),
    ], remainder="drop")

def create_pipeline(scaling_method, feature_selection_strategy, model_name, num_cols, cat_cols):
    steps = [("preprocess", make_preprocessor(scaling_method, num_cols, cat_cols, feature_selection_strategy))]
    
    # Add feature selection if specified (except Chi2 and ANOVA which are handled in preprocessor)
    if feature_selection_strategy != "none" and feature_selection_strategy != "L1" and feature_selection_strategy not in ["Chi2", "ANOVA"]:
        selector = FEATURE_SELECTION_STRATEGIES[feature_selection_strategy]["selector"]
        steps.append(("feature_selection", selector))
    
    model_config = MODELS_CONFIG[model_name]
    steps.append(("clf", model_config["estimator"]))
    
    return Pipeline(steps)

def get_param_grid(scaling_method, feature_selection_strategy, model_name, num_cols, cat_cols):
    param_grid = {}
    
    model_config = MODELS_CONFIG[model_name]
    param_grid.update(model_config["base_params"])
    
    # Add feature selection parameters with dynamic k values
    if feature_selection_strategy == "Chi2":
        cat_feature_count = sum([len(df[col].unique()) for col in cat_cols])
        param_grid["preprocess__cat__cat_sel__k"] = k_grid(cat_feature_count)
    elif feature_selection_strategy == "ANOVA":
        param_grid["preprocess__num__num_sel__k"] = k_grid(len(num_cols))
    elif feature_selection_strategy != "none" and feature_selection_strategy != "L1":
        total_feature_count = len(num_cols) + sum([len(df[col].unique()) for col in cat_cols])
        param_grid["feature_selection__k"] = k_grid(total_feature_count)
    
    # Handle L1 regularization for linear models
    if feature_selection_strategy == "L1" and model_name in ["LogisticRegression"]:
        l1_params = FEATURE_SELECTION_STRATEGIES["L1"]["params"]
        param_grid.update(l1_params)
        param_grid["clf__penalty"] = ["l1"]
        param_grid["clf__solver"] = ["saga"] 
    
    return param_grid


def fit_and_eval(X, y, estimator, param_grid, outer_cv, return_best_params=False):
    grid_search = GridSearchCV(estimator, param_grid=param_grid, scoring="f1_macro", 
                             cv=inner_cv, n_jobs=2, refit=True)  # Reduce parallel jobs
    
    cv_results = cross_validate(grid_search, X, y, cv=outer_cv, scoring=SCORING, 
                              return_train_score=False, n_jobs=2)  # Reduce parallel jobs
    
    scores = {
        "f1": cv_results["test_f1_macro"].tolist(),
        "acc": cv_results["test_accuracy"].tolist(),
        "auc": cv_results["test_roc_auc"].tolist(),
        "auprc": cv_results["test_average_precision"].tolist()
    }
    
    if return_best_params:
        # Fit on full data to get best parameters
        grid_search.fit(X, y)
        return scores, grid_search.best_params_
    
    return scores

def eval_with_fixed_params(X, y, estimator, outer_cv):
    """Evaluate estimator with fixed parameters (no hyperparameter tuning)"""
    cv_results = cross_validate(estimator, X, y, cv=outer_cv, scoring=SCORING, 
                              return_train_score=False, n_jobs=2)  # Reduce parallel jobs
    
    scores = {
        "f1": cv_results["test_f1_macro"].tolist(),
        "acc": cv_results["test_accuracy"].tolist(),
        "auc": cv_results["test_roc_auc"].tolist(),
        "auprc": cv_results["test_average_precision"].tolist()
    }
    
    return scores

def summarize(scores):
    """Calculate mean and std for all scores"""
    def _m(xs): return float(np.mean(xs)) if xs else np.nan
    def _s(xs): return float(np.std(xs, ddof=1)) if len(xs) > 1 else np.nan
    return {k: _m(v) for k, v in scores.items()} | {f"{k}_std": _s(v) for k, v in scores.items()}

def train_strategy(X, y, fs_name, scaling_method, feature_selection_strategy, model_name, nested=True, return_best_params=False):
    """Train model with specified strategy"""
    print(f"  {scaling_method} + {feature_selection_strategy} + {model_name} ({'Nested 3x3' if nested else 'CV=5'})")
    
    num_cols, cat_cols = split_cols(FEATURE_SETS[fs_name], num_features, cat_features)
    
    pipe = create_pipeline(scaling_method, feature_selection_strategy, model_name, num_cols, cat_cols)
    param_grid = get_param_grid(scaling_method, feature_selection_strategy, model_name, num_cols, cat_cols)
    
    cv_outer = outer_cv if nested else single_cv
    
    if return_best_params:
        scores, best_params = fit_and_eval(X, y, pipe, param_grid, cv_outer, return_best_params=True)
    else:
        scores = fit_and_eval(X, y, pipe, param_grid, cv_outer)
    
    stat = summarize(scores)
    print(f"    F1={stat['f1']:.4f}±{stat['f1_std']:.4f} | AUPRC={stat['auprc']:.4f}±{stat['auprc_std']:.4f}")
    
    result = {
        "FeatureSet": fs_name,
        "Scaling": scaling_method,
        "FeatureSelection": feature_selection_strategy,
        "Model": model_name,
        "Accuracy": round(stat["acc"], 4),
        "Accuracy_std": round(stat["acc_std"], 4),
        "F1_macro": round(stat["f1"], 4),
        "F1_macro_std": round(stat["f1_std"], 4),
        "AUPRC": round(stat["auprc"], 4) if not np.isnan(stat["auprc"]) else None,
        "AUPRC_std": round(stat["auprc_std"], 4) if not np.isnan(stat["auprc_std"]) else None
    }
    
    if return_best_params:
        return result, best_params
    return result

def compare_strategies(results_df, fs_name, model_name):
    model_results = results_df[(results_df["FeatureSet"] == fs_name) & 
                              (results_df["Model"] == model_name)]
    
    if len(model_results) == 0:
        return []
    
    comparisons = []
    
    # Compare scaling methods
    for scaling in SCALING_METHODS.keys():
        scaling_results = model_results[model_results["Scaling"] == scaling]
        if len(scaling_results) == 0:
            continue
            
        # Compare feature selection strategies within each scaling method
        fs_strategies = scaling_results["FeatureSelection"].unique()
        for i, fs1 in enumerate(fs_strategies):
            for fs2 in fs_strategies[i+1:]:
                result1 = scaling_results[scaling_results["FeatureSelection"] == fs1]
                result2 = scaling_results[scaling_results["FeatureSelection"] == fs2]
                
                if len(result1) > 0 and len(result2) > 0:
                    f1_1 = result1["F1_macro"].iloc[0]
                    f1_2 = result2["F1_macro"].iloc[0]
                    f1_std_1 = result1["F1_macro_std"].iloc[0]
                    f1_std_2 = result2["F1_macro_std"].iloc[0]
                    
                    auprc_1 = result1["AUPRC"].iloc[0]
                    auprc_2 = result2["AUPRC"].iloc[0]
                    auprc_std_1 = result1["AUPRC_std"].iloc[0]
                    auprc_std_2 = result2["AUPRC_std"].iloc[0]
                    
                    # Determine winner based on F1 score
                    mean_diff = f1_1 - f1_2
                    std_diff = np.sqrt(f1_std_1**2 + f1_std_2**2)
                    
                    if abs(mean_diff) <= std_diff:
                        winner = "Tie"
                    elif mean_diff > 0:
                        winner = f"{scaling}_{fs1}"
                    else:
                        winner = f"{scaling}_{fs2}"
                    
                    comparisons.append({
                        "FeatureSet": fs_name,
                        "Model": model_name,
                        "Strategy1": f"{scaling}_{fs1}",
                        "Strategy2": f"{scaling}_{fs2}",
                        "Strategy1_F1": f1_1,
                        "Strategy1_F1_std": f1_std_1,
                        "Strategy2_F1": f1_2,
                        "Strategy2_F1_std": f1_std_2,
                        "Strategy1_AUPRC": f"{auprc_1:.4f}±{auprc_std_1:.4f}" if auprc_1 is not None else "N/A",
                        "Strategy2_AUPRC": f"{auprc_2:.4f}±{auprc_std_2:.4f}" if auprc_2 is not None else "N/A",
                        "Winner": winner,
                        "Mean_Diff": round(mean_diff, 4),
                        "Std_Diff": round(std_diff, 4)
                    })
    
    return comparisons

print("=== Step 2: Two-Stage Comprehensive Strategy Testing ===")

# Stage 1: Hyperparameter tuning on subsampled data
print("\n=== Stage 1: Hyperparameter Tuning (Subsampled Data) ===")
tuning_results = []
best_params_storage = {}

plan = [("combined", True), ("lifestyle_only", False), ("biomedical_only", False)]

for fs_name, use_nested in plan:
    cols = FEATURE_SETS[fs_name]
    Xfs = X_train_full[cols]
    print(f"\nFeature set: {fs_name} [{'Nested 3x3' if use_nested else 'CV=5'}]")
    
    for scaling_method in SCALING_METHODS.keys():
        print(f"\nScaling: {scaling_method}")
        
        for model_name, model_config in MODELS_CONFIG.items():
            print(f"\nModel: {model_name}")
            
            for feature_selection_strategy in model_config["feature_selection"]:
                try:
                    result, best_params = train_strategy(Xfs, y_train, fs_name, scaling_method, 
                                                       feature_selection_strategy, model_name, 
                                                       nested=use_nested, return_best_params=True)
                    tuning_results.append(result)
                    
                    # Store best parameters for Stage 2
                    strategy_key = f"{fs_name}_{scaling_method}_{feature_selection_strategy}_{model_name}"
                    best_params_storage[strategy_key] = best_params
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

# Stage 2: Final evaluation on full dataset with best parameters
print("\n=== Stage 2: Final Evaluation (Full Dataset) ===")

# Restore full dataset
X_full = X_full_original
y = y_original
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Stage 2: Using full dataset with {len(y)} samples")
print(f"Class distribution: {np.bincount(y)}")

all_results = []

for fs_name, use_nested in plan:
    cols = FEATURE_SETS[fs_name]
    Xfs = X_train_full[cols]
    print(f"\nFeature set: {fs_name} [Full Dataset]")
    
    for scaling_method in SCALING_METHODS.keys():
        print(f"\nScaling: {scaling_method}")
        
        for model_name, model_config in MODELS_CONFIG.items():
            print(f"\nModel: {model_name}")
            
            for feature_selection_strategy in model_config["feature_selection"]:
                try:
                    # Use best parameters from Stage 1
                    strategy_key = f"{fs_name}_{scaling_method}_{feature_selection_strategy}_{model_name}"
                    if strategy_key in best_params_storage:
                        # Create pipeline with best parameters
                        num_cols, cat_cols = split_cols(FEATURE_SETS[fs_name], num_features, cat_features)
                        pipe = create_pipeline(scaling_method, feature_selection_strategy, model_name, num_cols, cat_cols)
                        pipe.set_params(**best_params_storage[strategy_key])
                        
                        # Evaluate on full dataset with fixed parameters (no GridSearch)
                        cv_outer = outer_cv if use_nested else single_cv
                        scores = eval_with_fixed_params(Xfs, y_train, pipe, cv_outer)
                        stat = summarize(scores)
                        print(f"    F1={stat['f1']:.4f}±{stat['f1_std']:.4f} | AUPRC={stat['auprc']:.4f}±{stat['auprc_std']:.4f}")
                        
                        result = {
                            "FeatureSet": fs_name,
                            "Scaling": scaling_method,
                            "FeatureSelection": feature_selection_strategy,
                            "Model": model_name,
                            "Accuracy": round(stat["acc"], 4),
                            "Accuracy_std": round(stat["acc_std"], 4),
                            "F1_macro": round(stat["f1"], 4),
                            "F1_macro_std": round(stat["f1_std"], 4),
                            "AUPRC": round(stat["auprc"], 4) if not np.isnan(stat["auprc"]) else None,
                            "AUPRC_std": round(stat["auprc_std"], 4) if not np.isnan(stat["auprc_std"]) else None
                        }
                        all_results.append(result)
                    else:
                        print(f"  No best parameters found, skipping...")
                        
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

# Save hyperparameter tuning results
if tuning_results:
    tuning_df = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(RESULTS_DIR, "rq1a_hyperparameter_tuning_results.csv")
    tuning_df.to_csv(tuning_csv_path, index=False, encoding="utf-8-sig")
    print(f"Hyperparameter tuning results saved to: {tuning_csv_path}")

# Save final evaluation results
results_df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "rq1a_comprehensive_strategy_test.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Results saved to: {csv_path}")

# Generate strategy comparisons
comparison_results = []
for fs_name in FEATURE_SETS.keys():
    for model_name in MODELS_CONFIG.keys():
        comparisons = compare_strategies(results_df, fs_name, model_name)
        comparison_results.extend(comparisons)

if comparison_results:
    comparison_df = pd.DataFrame(comparison_results)
    comparison_csv_path = os.path.join(RESULTS_DIR, "rq1a_strategy_comparisons.csv")
    comparison_df.to_csv(comparison_csv_path, index=False, encoding="utf-8-sig")
    print(f"Strategy comparisons saved to: {comparison_csv_path}")

# ==== Visualization ====
def plot_strategy_comparison(results_df, save_path):
    """Plot strategy comparison heatmap"""
    if len(results_df) == 0:
        return
    
    # Create pivot table for F1 scores
    pivot_data = results_df.pivot_table(
        values="F1_macro", 
        index=["FeatureSet", "Model"], 
        columns=["Scaling", "FeatureSelection"], 
        aggfunc="mean"
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    feature_sets = results_df["FeatureSet"].unique()
    
    for i, fs in enumerate(feature_sets):
        ax = axes[i]
        fs_data = pivot_data.loc[fs] if fs in pivot_data.index else None
        
        if fs_data is not None and not fs_data.empty:
            im = ax.imshow(fs_data.values, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(fs_data.columns)))
            ax.set_xticklabels([f"{col[0]}_{col[1]}" for col in fs_data.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(fs_data.index)))
            ax.set_yticklabels(fs_data.index)
            ax.set_title(f"Strategy Performance - {fs}")
            
            plt.colorbar(im, ax=ax, label="F1 Score")
            
            for j in range(len(fs_data.index)):
                for k in range(len(fs_data.columns)):
                    text = ax.text(k, j, f"{fs_data.iloc[j, k]:.3f}",
                                 ha="center", va="center", color="white" if fs_data.iloc[j, k] < 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if len(results_df) > 0:
    plot_path = os.path.join(PLOTS_DIR, "rq1a_strategy_comparison.png")
    plot_strategy_comparison(results_df, plot_path)
    print(f"Visualization saved to: {plot_path}")
