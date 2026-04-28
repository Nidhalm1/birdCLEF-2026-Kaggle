import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import time

SAVED_BASE_PATH = "saved/" # "/kaggle/input/datasets/emirhansagir/projmlsaved/saved/"

def xgboost_model(X_array=None, Y_encoded=None, groups=None, n_estimators=600, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8):
    print("\n/// XGBoost Model ///")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}, colsample_bytree={colsample_bytree}")
    if X_array is None:
        X_array = np.load(SAVED_BASE_PATH + "X.npy", allow_pickle=True)
        print(f"Loaded X_array shape: {X_array.shape}")
    if Y_encoded is None:
        Y_encoded = np.load(SAVED_BASE_PATH + "Y_encoded.npy", allow_pickle=True)
        print(f"Loaded Y_encoded shape: {Y_encoded.shape}")
    if groups is None:
        groups = np.load(SAVED_BASE_PATH + "groups.npy", allow_pickle=True)
        print(f"Loaded groups shape: {groups.shape}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]

    # Base XGBoost model
    base_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    # Wrap for multi-label
    model = MultiOutputClassifier(base_model, n_jobs=-1)

    print("Training XGBoost model...")
    model.fit(X_train, Y_train)

    # Probabilities
    Y_pred_proba_list = model.predict_proba(X_test)

    Y_pred_proba = np.stack([
        proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0])
        for proba in Y_pred_proba_list
    ], axis=1)

    # AUC computation
    auc_scores = []
    valid_classes = []

    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_pred_proba[:, i]

        if len(np.unique(y_true)) > 1:
            auc_scores.append(roc_auc_score(y_true, y_pred))
            valid_classes.append(i)

    if len(auc_scores) > 0:
        auc_score = np.mean(auc_scores)
    else:
        auc_score = 0.5

    return model, auc_score, X_test, Y_test


def xboost_model_tests():
    n_estimators = [100, 300, 600]
    max_depth = [4, 6, 8]
    learning_rate = [0.01, 0.05]
    subsample = [0.8, 1.0]
    colsample_bytree = [0.8, 1.0]

    total_combinations = len(n_estimators) * len(max_depth) * len(learning_rate) * len(subsample) * len(colsample_bytree)
    print("/// XGBoost Hyperparameter Tests ///")
    print(f"Testing combinations of: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}, colsample_bytree={colsample_bytree}")
    print(f"Total combinations to test: {total_combinations}")
    counter = 0

    results = []

    best_model = None
    best_score = 0
    best_params = {}

    print("Starting XGBoost hyperparameter tests...")
    print("loading data...")

    X_array = np.load(SAVED_BASE_PATH + "X.npy", allow_pickle=True)
    Y_encoded = np.load(SAVED_BASE_PATH + "Y_encoded.npy", allow_pickle=True)
    groups = np.load(SAVED_BASE_PATH + "groups.npy", allow_pickle=True)

    for n in n_estimators:
        for d in max_depth:
            for lr in learning_rate:
                for ss in subsample:
                    for cs in colsample_bytree:
                        counter += 1
                        print("\n\n----------------------------------------")
                        print(f"Testing combination {counter} of {total_combinations}")
                        print(f"Testing: n={n}, depth={d}, lr={lr}, subsample={ss}, colsample={cs}")

                        start_time = time.time()

                        model, score, _, _ = xgboost_model(
                            X_array=X_array,
                            Y_encoded=Y_encoded,
                            groups=groups,
                            n_estimators=n,
                            max_depth=d,
                            learning_rate=lr,
                            subsample=ss,
                            colsample_bytree=cs
                        )

                        duration = time.time() - start_time

                        print(f"Score: {score} | Time: {duration:.2f}s")

                        run_result = {
                            "n_estimators": n,
                            "max_depth": d,
                            "learning_rate": lr,
                            "subsample": ss,
                            "colsample_bytree": cs,
                            "auc_score": score,
                            "train_time_sec": duration
                        }

                        results.append(run_result)

                        if score > best_score:
                            best_score = score
                            best_params = run_result
                            best_model = model
                        print(f"Current best score: {best_score} with params: {best_params}")

    print(f"\nBest score: {best_score}")
    print(f"Best params: {best_params}")

    return best_model, best_params, best_score, results