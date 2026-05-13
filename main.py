from logisticRegression import logistic_regression_model_tests
from svm import svm_model_tests
from randomForestClassifier import randomForest_model_tests
from xgboostClassifier import xgboost_model, xboost_model_tests
from audioCharge import *
from visualize import plot_results

import joblib
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from paths import SAVED_BASE_PATH, OUTPUT_BASE_PATH




def ensure_output_dir(rest_path):
    os.makedirs(OUTPUT_BASE_PATH + rest_path, exist_ok=True)


def test_model(name, X_train, X_test, Y_train, Y_test):
    best_model, best_params, best_score, results = None, None, None, None
    if name == "logistic_regression":
        best_model, best_params, best_score, results = logistic_regression_model_tests(X_train, X_test, Y_train, Y_test)
    elif name == "svm":
        best_model, best_params, best_score, results = svm_model_tests(X_train, X_test, Y_train, Y_test)
    elif name == "random_forest":
        best_model, best_params, best_score, results = randomForest_model_tests(X_train, X_test, Y_train, Y_test)
    elif name == "xgboost":
        best_model, best_params, best_score, results = xboost_model_tests(X_train, X_test, Y_train, Y_test)
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
    X_array = np.load(SAVED_BASE_PATH + "X.npy", allow_pickle=True)
    print(f"Loaded X_array shape: {X_array.shape}")

    Y_encoded = np.load(SAVED_BASE_PATH + "Y_encoded.npy", allow_pickle=True)
    print(f"Loaded Y_encoded shape: {Y_encoded.shape}")

    groups = np.load(SAVED_BASE_PATH + "groups.npy", allow_pickle=True)
    print(f"Loaded groups shape: {groups.shape}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]

    start = time.time()
    test_start = None
    test_end = None
    times = []
    # model_names = ["logistic_regression", "svm"]# ["svm", "random_forest", "xgboost"] # ["logistic_regression", "svm", "random_forest", "xgboost"]

    for model_name in model_names:
        print(f"\n\n=== Testing {model_name.replace('_', ' ').title()} ===")
        test_start = time.time()
        test_model(model_name, X_train, X_test, Y_train, Y_test)
        test_end = time.time()
        model_time = test_end - test_start
        print(f"Time taken for {model_name}: {model_time:.2f}s")
        times.append(model_time)
    
    end = time.time()
    print(f"Total time taken: {end - start:.2f}s")
    print("Individual model times:")
    for model_name, model_time in zip(model_names, times):
        print(f"{model_name.replace('_', ' ').title()}: {model_time:.2f}s")

if __name__ == "__main__":
    test_models(["logistic_regression", "svm", "random_forest", "xgboost"])
    print("\n\n=== Generating Plots ===")
    plot_results()
