import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

def normalize(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    vmax = np.max(np.abs(v)) if np.max(np.abs(v)) != 0 else 1.0
    return np.abs(v) / vmax

def prepare_data(df):
    
    df = df[
        (df["ap_hi"].between(80, 240))
        & (df["ap_lo"].between(40, 180))
        & (df["ap_hi"] >= df["ap_lo"])
        & (df["height"].between(120, 220))
        & (df["weight"].between(30, 200))
    ].copy()
    df["age_years"] = (df["age"] / 365).astype(int)
    df["bmi"] = df["weight"] / ((df["height"]/100) ** 2)
    df = df[(df["bmi"].between(10, 60))].copy()
    y = df["cardio"].astype(int)
    feature_cols = ["age_years","gender","height","weight","ap_hi","ap_lo",
                    "cholesterol","gluc","smoke","alco","active","bmi"]
    X = df[feature_cols].copy()
    num_cols = ["age_years","height","weight","ap_hi","ap_lo","bmi"]
    cat_cols = ["gender","cholesterol","gluc","smoke","alco","active"]
    return X, y, num_cols, cat_cols

def plot_grouped_bar(df, out_path, title):
    
    import numpy as np, matplotlib.pyplot as plt
    x = np.arange(len(df))
    w = 0.25
    plt.figure(figsize=(10,6))
    plt.bar(x - w, df["LogisticRegression"].values, width=w, label="Logistic Regression")
    plt.bar(x,       df["RandomForest"].values,      width=w, label="Random Forest")
    plt.bar(x + w,   df["MLP"].values,               width=w, label="MLP")
    plt.xticks(x, df.index, rotation=45, ha="right")
    plt.ylabel("Normalized Importance")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to cardio_train.csv")
    ap.add_argument("--out", type=str, default=".", help="output directory")
    ap.add_argument("--downsample", type=int, default=0, help="optional row cap for speed (e.g., 5000)")
    ap.add_argument("--perm_repeats", type=int, default=5, help="permutation importance repeats for MLP")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sep = ";" if args.data.endswith(".csv") else None
    df = pd.read_csv(args.data, sep=sep)
    if args.downsample and args.downsample < len(df):
        df = df.sample(args.downsample, random_state=42)

    X, y, numeric_cols, cat_cols = prepare_data(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([("num", StandardScaler(), numeric_cols),
                             ("cat", "passthrough", cat_cols)])

    
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))]).fit(Xtr, ytr)
    mlp = Pipeline([("pre", pre), ("clf", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42, early_stopping=True))]).fit(Xtr, ytr)

    processed_feature_names = numeric_cols + cat_cols
    lr_imp = normalize(lr.named_steps["clf"].coef_[0])
    rf_imp = normalize(rf.named_steps["clf"].feature_importances_)
    perm = permutation_importance(mlp, Xte, yte, n_repeats=args.perm_repeats, random_state=42, scoring="f1")
    mlp_imp = normalize(perm.importances_mean)

    imp_df = pd.DataFrame({
        "feature": processed_feature_names,
        "LogisticRegression": lr_imp,
        "RandomForest": rf_imp,
        "MLP": mlp_imp
    }).set_index("feature")

    
    imp_table_path = os.path.join(args.out, "feature_importance_table.csv")
    imp_df.to_csv(imp_table_path)

    # Top 10 features
    top10 = imp_df.mean(axis=1).sort_values(ascending=False).head(10).index.tolist()
    imp_top10 = imp_df.loc[top10]


    plot_grouped_bar(imp_top10, os.path.join(args.out, "feature_importance.png"),
                     "Top-10 Feature Importance Comparison Across Models")

    rank_df = imp_df.rank(ascending=False, method="average")
    corr = rank_df.corr(method="pearson")


    import matplotlib.pyplot as plt, numpy as np
    plt.figure(figsize=(6,5))
    im = plt.imshow(corr.values)
    plt.xticks(np.arange(corr.shape[0]), corr.columns, rotation=45, ha="right")
    plt.yticks(np.arange(corr.shape[1]), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center")
    plt.title("Correlation of Feature Rankings Across Models")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "feature_rank_corr.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Summary
    rho_lr_rf, _ = spearmanr(imp_df["LogisticRegression"], imp_df["RandomForest"])
    rho_lr_mlp, _ = spearmanr(imp_df["LogisticRegression"], imp_df["MLP"])
    rho_rf_mlp, _ = spearmanr(imp_df["RandomForest"], imp_df["MLP"])
    with open(os.path.join(args.out, "rq1c_summary.txt"), "w") as f:
        f.write(f"Spearman rank correlations:\nLR-RF={rho_lr_rf:.3f}\nLR-MLP={rho_lr_mlp:.3f}\nRF-MLP={rho_rf_mlp:.3f}\n")

if __name__ == "__main__":
    main()