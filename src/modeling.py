import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.preprocessing import preprocess


# Helper Functions
def save_results(results: dict, filepath: str):
    """Save model results dictionary as JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Fit model, predict, and return metrics + parameters."""
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    results = {
        "params": model.get_params(),
        "MAE": mean_absolute_error(y_val, preds),
        "RMSE": np.sqrt(mean_squared_error(y_val, preds)),
        "R2": r2_score(y_val, preds),
    }
    return results


# Main Training Pipeline
def main(args):
    # Create results directories
    for folder in ["results/baselines", "results/candidates", "results/final"]:
        os.makedirs(folder, exist_ok=True)

    # Load datasets
    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    train_sample = pd.read_csv(args.train_sample)
    val_sample = pd.read_csv(args.val_sample)
    test = pd.read_csv(args.test)

    # Preprocess
    train = preprocess(train)
    val = preprocess(val)
    train_sample = preprocess(train_sample)
    val_sample = preprocess(val_sample)
    test = preprocess(test)

    train_X, train_y = train.drop("trip_duration_transformed", axis=1), train["trip_duration_transformed"]
    val_X, val_y = val.drop("trip_duration_transformed", axis=1), val["trip_duration_transformed"]
    train_sample_X, train_sample_y = train_sample.drop("trip_duration_transformed", axis=1), train_sample["trip_duration_transformed"]
    val_sample_X, val_sample_y = val_sample.drop("trip_duration_transformed", axis=1), val_sample["trip_duration_transformed"]
    test_X, test_y = test.drop("trip_duration_transformed", axis=1), test["trip_duration_transformed"]

    # 1. Baselines
    baseline_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest_Default": RandomForestRegressor(random_state=42),
    }

    baseline_results = {}
    for name, model in baseline_models.items():
        metrics = evaluate_model(model, train_sample_X, train_sample_y, val_sample_X, val_sample_y)
        baseline_results[name] = metrics
        scores = {k: baseline_results[name][k] for k in ["MAE", "RMSE", "R2"]}
        print(f"[Baseline] {name}: {scores}")

    save_results(baseline_results, "results/baselines/baseline_results.json")

    # 2. Candidate Models
    candidate_models = {
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=200, max_depth=-1, learning_rate=0.1, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostRegressor(n_estimators=200, depth=6, learning_rate=0.1, random_state=42, verbose=0),
    }

    candidate_results = {}
    for name, model in candidate_models.items():
        metrics = evaluate_model(model, train_sample_X, train_sample_y, val_sample_X, val_sample_y)
        candidate_results[name] = metrics
        scores = {k: candidate_results[name][k] for k in ["MAE", "RMSE", "R2"]}
        print(f"[Candidate] {name}: {scores}")

    save_results(candidate_results, "results/candidates/candidate_results.json")

    # 3. Final Model Selection
    print("\n[Final] Selecting best model based on validation performance...")

    # Combine all results (baselines + candidates)
    all_results = {**baseline_results, **candidate_results}

    # Pick model with highest R2
    best_model_name = max(all_results, key=lambda m: all_results[m]["R2"])
    print(f"Best model selected: {best_model_name}")

    # Retrieve the corresponding model object
    best_model = None
    if best_model_name in baseline_models:
        best_model = baseline_models[best_model_name]
    else:
        best_model = candidate_models[best_model_name]

    # Train on full (train + val)
    full_X = pd.concat([train_X, val_X], axis=0)
    full_y = pd.concat([train_y, val_y], axis=0)

    best_model.fit(full_X, full_y)

    # Save final model
    final_model_path = f"results/final/{best_model_name}_final.pkl"
    joblib.dump(best_model, final_model_path)

    # 4. Final Testing
    print(f"\n[Final] Evaluating {best_model_name} on test set...")
    y_test_preds = best_model.predict(test_X)

    def final_res(actual, pred):
        return {
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "R2": r2_score(actual, pred),
        }

    final_result_test = final_res(test_y, y_test_preds)
    print(f"\nFinal Test Results: {final_result_test}")

    save_results(final_result_test, "results/final/testing_results.json")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Trip Duration - Modeling Pipeline")
    parser.add_argument("--train", type=str, default="data/raw/split/train.csv", help="Path to training CSV")
    parser.add_argument("--val", type=str, default="data/raw/split/val.csv", help="Path to validation CSV")
    parser.add_argument("--train_sample", type=str, default="data/raw/split_sample/train.csv", help="Path to sample training CSV")
    parser.add_argument("--val_sample", type=str, default="data/raw/split_sample/val.csv", help="Path to sample validation CSV")
    parser.add_argument("--test", type=str,default="data/raw/split/test.csv", help="Path to test CSV")

    args = parser.parse_args()
    main(args)
