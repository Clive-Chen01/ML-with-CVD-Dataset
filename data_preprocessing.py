import pandas as pd
import numpy as np


def load_and_preprocess_data(data_path="cardio_train.csv", save_processed=True, processed_path="cardio_processed.csv"):
    """
    Args:
        data_path (str): original data file path
        save_processed (bool): whether to save processed data
        processed_path (str): path to save processed data
        
    Returns:
        tuple: (df, target, num_features, cat_features, feature_sets)
            - df: processed data frame
            - target: target variable name
            - num_features: numerical features list
            - cat_features: categorical features list
            - feature_sets: feature sets dictionary
    """
    
    df = pd.read_csv(data_path, sep=';') if data_path.endswith(".csv") else pd.read_csv(data_path)
    
    # data cleaning 
    # convert age from days to years
    df["age_years"] = (df["age"] / 365.25).round(2)
    # BMI（kg/m^2）
    df["height_m"] = df["height"] / 100.0
    df["bmi"] = df["weight"] / (df["height_m"] ** 2)
    
    # filter outliers
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 180)]
    df = df[df["ap_hi"] >= df["ap_lo"]]
    
    df = df[(df["height"] >= 120) & (df["height"] <= 220)]
    df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]
    
    df = df[(df["bmi"] >= 10) & (df["bmi"] <= 60)]
    
    # target and feature definition
    target = "cardio"
    num_features = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]
    cat_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    
    df = df.drop(columns=["age", "height_m"], errors="ignore")
    df = df.dropna(subset=num_features + cat_features + [target]).copy()
    df[target] = df[target].astype(int)
    
    LIFESTYLE = ["smoke", "alco", "active"]
    BIOMED = ["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "bmi", "height", "weight"]
    COMBINED = LIFESTYLE + BIOMED + ["gender"]
    
    feature_sets = {
        "lifestyle_only": LIFESTYLE,
        "biomedical_only": BIOMED,
        "combined": COMBINED
    }
    
    if save_processed:
        df.to_csv(processed_path, index=False, encoding="utf-8-sig")
        print(f"processed data saved to: {processed_path}")
    
    return df, target, num_features, cat_features, feature_sets


def load_processed_data(processed_path="cardio_processed.csv"):

    df = pd.read_csv(processed_path)
    
    target = "cardio"
    num_features = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]
    cat_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    
    LIFESTYLE = ["smoke", "alco", "active"]
    BIOMED = ["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "bmi", "height", "weight"]
    COMBINED = LIFESTYLE + BIOMED + ["gender"]
    
    feature_sets = {
        "lifestyle_only": LIFESTYLE,
        "biomedical_only": BIOMED,
        "combined": COMBINED
    }
    
    return df, target, num_features, cat_features, feature_sets

if __name__ == "__main__":
    df, target, num_features, cat_features, feature_sets = load_and_preprocess_data()
    
    for name, features in feature_sets.items():
        print(f"  {name}: {features}")
