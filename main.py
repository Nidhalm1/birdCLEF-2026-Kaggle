from logger import setup_logging, log_section, close_logging
LOG_PATH = setup_logging("main")  # creates outputs/logs/main_<timestamp>.log

from logisticRegression import logistic_regression_model_tests
from svm import svm_model_tests
from randomForestClassifier import randomForest_model_tests
from xgboostClassifier import xgboost_model_tests
from audioCharge import *
from visualize import plot_results

import joblib
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from paths import SAVED_BASE_PATH, SAVED_OUTPUT_PATH, OUTPUT_BASE_PATH




def ensure_output_dir(rest_path):
    os.makedirs(OUTPUT_BASE_PATH + rest_path, exist_ok=True)


def resolve_saved_path():
    """Pick the saved/ directory to load from.
    Prefer SAVED_OUTPUT_PATH (freshly built), fall back to SAVED_BASE_PATH (pre-saved).
    """
    candidate1 = SAVED_OUTPUT_PATH
    candidate2 = SAVED_BASE_PATH
    if os.path.exists(os.path.join(candidate1, "X.npy")):
        print(f"Loading from SAVED_OUTPUT_PATH: {candidate1}")
        return candidate1
    elif os.path.exists(os.path.join(candidate2, "X.npy")):
        print(f"Loading from SAVED_BASE_PATH: {candidate2}")
        return candidate2
    else:
        raise FileNotFoundError(
            f"X.npy not found in either {candidate1} or {candidate2}. "
            f"Run audioCharge.py first to build the dataset."
        )


def test_model(name, X_train, X_test, Y_train, Y_test):
    best_model, best_params, best_score, results = None, None, None, None
    if name == "logistic_regression":
        best_model, best_params, best_score, results = logistic_regression_model_tests(X_train, X_test, Y_train, Y_test)
    elif name == "svm":
        best_model, best_params, best_score, results = svm_model_tests(X_train, X_test, Y_train, Y_test)
    elif name == "random_forest":
        best_model, best_params, best_score, results = randomForest_model_tests(X_train, X_test, Y_train, Y_test)
    elif name == "xgboost":
        best_model, best_params, best_score, results = xgboost_model_tests(X_train, X_test, Y_train, Y_test)
    else:
        print(f"Unknown model name: {name}")
        return None, None, None, None

    print(f"Best {name.capitalize()} Parameters:", best_params)
    print(f"Best {name.capitalize()} Score:", best_score)


    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    ensure_output_dir("results/")
    result_file_path = f"{OUTPUT_BASE_PATH}results/{name}_results.csv"
    df.to_csv(result_file_path, index=False)

    print(f"Results saved to {result_file_path}")

    # Save model
    ensure_output_dir("models/")
    model_path = f"{OUTPUT_BASE_PATH}models/{name}_best_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"Best model saved to {model_path}")


def test_models(model_names):
    log_section("Loading dataset")
    saved_path = resolve_saved_path()

    X_array = np.load(os.path.join(saved_path, "X.npy"), allow_pickle=True)
    print(f"Loaded X_array shape: {X_array.shape}")

    Y_encoded = np.load(os.path.join(saved_path, "Y_encoded.npy"), allow_pickle=True)
    print(f"Loaded Y_encoded shape: {Y_encoded.shape}")

    groups = np.load(os.path.join(saved_path, "groups.npy"), allow_pickle=True)
    print(f"Loaded groups shape: {groups.shape}")

    log_section("Splitting (group-aware)")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]
    print(f"Train samples: {X_train.shape[0]}  Test samples: {X_test.shape[0]}")
    print(f"Train groups:  {len(set(groups[train_idx]))}  Test groups: {len(set(groups[test_idx]))}")

    start = time.time()
    times = []

    for model_name in model_names:
        log_section(f"Testing {model_name.replace('_', ' ').title()}")
        test_start = time.time()
        test_model(model_name, X_train, X_test, Y_train, Y_test)
        test_end = time.time()
        model_time = test_end - test_start
        print(f"\n>>> Time taken for {model_name}: {model_time:.2f}s ({model_time/60:.1f} min)")
        times.append(model_time)

    end = time.time()
    log_section("FINAL SUMMARY")
    print(f"Total time taken: {end - start:.2f}s  ({(end-start)/60:.1f} min, {(end-start)/3600:.2f} h)")
    print("Individual model times:")
    for model_name, model_time in zip(model_names, times):
        print(f"  {model_name.replace('_', ' ').title():25s}: {model_time:8.2f}s  ({model_time/60:6.1f} min)")


if __name__ == "__main__":
    try:
        test_models(["logistic_regression", "svm", "random_forest", "xgboost"])
        log_section("Generating Plots")
        plot_results()
    finally:
        close_logging()
