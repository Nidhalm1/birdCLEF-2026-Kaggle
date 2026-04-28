import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def logistic_regression_model(X_array=None, Y_encoded=None, groups=None, C=1.0):

    print("\n/// Logistic Regression Model ///")
    print(f"Parameters: C={C}")

    if X_array is None:
        X_array = np.load("saved/X.npy", allow_pickle=True)
        print(f"Loaded X_array shape: {X_array.shape}")
    if Y_encoded is None:
        Y_encoded = np.load("saved/Y_encoded.npy", allow_pickle=True)
        print(f"Loaded Y_encoded shape: {Y_encoded.shape}")
    if groups is None:
        groups = np.load("saved/groups.npy", allow_pickle=True)
        print(f"Loaded groups shape: {groups.shape}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]

    # Scaling (IMPORTANT)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MultiOutputClassifier(
        LogisticRegression(
            C=C,
            max_iter=1000,
            n_jobs=-1
        ),
        n_jobs=-1
    )

    print("Training Logistic Regression model...")
    model.fit(X_train, Y_train)

    # Probabilities
    Y_pred_proba_list = model.predict_proba(X_test)

    # Same format as RF/XGB
    Y_pred_proba = np.stack([
        proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0])
        for proba in Y_pred_proba_list
    ], axis=1)

    # AUC
    auc_scores = []
    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_pred_proba[:, i]

        if len(np.unique(y_true)) > 1:
            auc_scores.append(roc_auc_score(y_true, y_pred))

    auc_score = np.mean(auc_scores) if len(auc_scores) > 0 else 0.5

    return model, auc_score, X_test, Y_test


def logistic_regression_model_tests():
    C_values = [0.01, 0.1, 1.0, 10.0]

    total_combinations = len(C_values)
    print("/// Logistic Regression Hyperparameter Tests ///")
    print(f"Testing C values: {C_values}")
    print(f"Total combinations: {total_combinations}")

    results = []
    best_model = None
    best_score = 0
    best_params = {}

    X_array = np.load("saved/X.npy", allow_pickle=True)
    Y_encoded = np.load("saved/Y_encoded.npy", allow_pickle=True)
    groups = np.load("saved/groups.npy", allow_pickle=True)

    for i, C in enumerate(C_values, 1):
        print("\n----------------------------------------")
        print(f"Test {i}/{total_combinations} | C={C}")

        start_time = time.time()

        model, score, _, _ = logistic_regression_model(
            X_array=X_array,
            Y_encoded=Y_encoded,
            groups=groups,
            C=C
        )

        duration = time.time() - start_time

        print(f"Score: {score} | Time: {duration:.2f}s")

        run_result = {
            "C": C,
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