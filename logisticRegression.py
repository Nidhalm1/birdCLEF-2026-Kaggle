import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def logistic_regression_model(X_train, X_test, Y_train, Y_test, C=1.0, max_iter=1000):

    print("\n/// Logistic Regression Model ///")
    print(f"Parameters: C={C}, max_iter={max_iter}")

    valid_labels = []
    for i in range(Y_train.shape[1]):
        if len(np.unique(Y_train[:, i])) > 1:
            valid_labels.append(i)

    if len(valid_labels) == 0:
        print("No valid labels found (all single-class).")
        return None, 0.5, X_test, Y_test

    Y_train_filtered = Y_train[:, valid_labels]
    Y_test_filtered = Y_test[:, valid_labels]

    print(f"Using {len(valid_labels)}/{Y_train.shape[1]} valid labels")

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MultiOutputClassifier(
        LogisticRegression(
            C=C,
            max_iter=max_iter
        ),
        n_jobs=-1
    )

    print("Training Logistic Regression model...")
    model.fit(X_train, Y_train_filtered)

    # Probabilities
    Y_pred_proba_list = model.predict_proba(X_test)

    Y_pred_proba_filtered = np.stack([
        proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0])
        for proba in Y_pred_proba_list
    ], axis=1)

    Y_pred_proba = np.zeros((X_test.shape[0], Y_train.shape[1]))
    Y_pred_proba[:, valid_labels] = Y_pred_proba_filtered

    # AUC
    auc_scores = []
    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_pred_proba[:, i]

        if len(np.unique(y_true)) > 1:
            auc_scores.append(roc_auc_score(y_true, y_pred))

    auc_score = np.mean(auc_scores) if len(auc_scores) > 0 else 0.5

    return model, auc_score, X_test, Y_test


def logistic_regression_model_tests(X_train, X_test, Y_train, Y_test):
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    max_iter_values = [500, 1000, 5000, 10000]

    total_combinations = len(C_values) * len(max_iter_values)
    print("/// Logistic Regression Hyperparameter Tests ///")
    print(f"Testing C values: {C_values}")
    print(f"Testing max_iter values: {max_iter_values}")
    print(f"Total combinations: {total_combinations}")

    results = []
    best_model = None
    best_score = 0
    best_params = {}

    for i, C in enumerate(C_values, 1):
        for j, max_iter in enumerate(max_iter_values, 1):
            print("\n----------------------------------------")
            print(f"Test {i}/{total_combinations} | C={C}, max_iter={max_iter}")

            start_time = time.time()

            model, score, _, _ = logistic_regression_model(
                X_train, X_test, Y_train, Y_test,
                C=C,
                max_iter=max_iter
            )

            duration = time.time() - start_time

            print(f"Score: {score} | Time: {duration:.2f}s")

            run_result = {
                "C": C,
                "max_iter": max_iter,
                "auc_score": score,
                "train_time_sec": duration
            }

            results.append(run_result)

            if score > best_score:
                best_score = score
                best_params = run_result
                best_model = model

    print(f"\nBest score: {best_score}")
    print(f"Best params: {best_params}")

    return best_model, best_params, best_score, results
