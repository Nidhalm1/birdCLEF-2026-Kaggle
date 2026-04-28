from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import pandas as pd

SAVED_BASE_PATH = "/kaggle/input/datasets/emirhansagir/projmlsaved/saved/"
OUTPUT_BASE_PATH = "/kaggle/working/"


def randomForest(X_array=None, Y_encoded=None, groups=None, n_estimators=600, max_depth=6, class_weight="balanced", max_features="sqrt", min_samples_split=2):
    print("\n/// RandomForest Model ///")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, class_weight={class_weight}, max_features={max_features}, min_samples_split={min_samples_split}")
    if X_array is None:
        X_array = np.load(SAVED_BASE_PATH + "X.npy", allow_pickle=True)
        print(f"Loaded X_array shape: {X_array.shape}")
    if Y_encoded is None:
        Y_encoded = np.load(SAVED_BASE_PATH + "Y_encoded.npy", allow_pickle=True)
        print(f"Loaded Y_encoded shape: {Y_encoded.shape}")
    if groups is None:
        groups = np.load(SAVED_BASE_PATH + "groups.npy", allow_pickle=True)
        print(f"Loaded groups shape: {groups.shape}")

    # Garder seulement les 26 premiers éléments de chaque feature
    #X_array = np.array([x[] for x in X_array])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]
    

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        max_features=max_features,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, Y_train)

    # Probabilités
    Y_pred_proba_list = model.predict_proba(X_test)

    Y_pred_proba = np.stack([
        proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0])
        for proba in Y_pred_proba_list
    ], axis=1)

    # AUC SAFE
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
    #print("Y_encoded") [[1 0 0 ... 0 0 0][0 1 0 ... 0 0 0]]
    #print(Y_pred_proba) [[0.1322108  0.01       0.0295151  ... 0.005      0.         0.015     ][0.16297093 0.02       0.00693374 ... 0.02       0.01       0.015     ]
    # print("Y_pred_proba_list")
    # print(Y_pred_proba_list) # array([[1.        , 0.        ],[1.        , 0.        ],[0.995     , 0.005     ],...,[0.98058876, 0.01941124]], shape=(872, 2)), array([[0.995     , 0.005     ],
    # chaque array est une espace et et à lintereieur c'est des segment y en 872


    return model, auc_score, X_test, Y_test


def randomForest_model_tests():
    n_estimators = [100, 300, 600, 800]
    max_depth = [4, 6, 8, 10]
    class_weight = ["balanced"]
    max_features = ["sqrt", "log2"]
    min_samples_split = [2, 5, 10]

    total_combinations = len(n_estimators) * len(max_depth) * len(class_weight) * len(max_features) * len(min_samples_split)
    print("/// RandomForest Hyperparameter Tests ///")
    print(f"Testing combinations of: n_estimators={n_estimators}, max_depth={max_depth}, class_weight={class_weight}, max_features={max_features}, min_samples_split={min_samples_split}")
    print(f"Total combinations to test: {total_combinations}")
    counter = 0

    results = []

    best_model = None
    best_score = 0
    best_params = {}

    print("Starting RandomForest hyperparameter tests...")
    print("loading data...")

    X_array = np.load(SAVED_BASE_PATH + "X.npy", allow_pickle=True)
    Y_encoded = np.load(SAVED_BASE_PATH + "Y_encoded.npy", allow_pickle=True)
    groups = np.load(SAVED_BASE_PATH + "groups.npy", allow_pickle=True)

    for n in n_estimators:
        for d in max_depth:
            for cw in class_weight:
                for mf in max_features:
                    for mss in min_samples_split:
                        counter += 1
                        print("\n\n----------------------------------------")
                        print(f"Testing combination {counter} of {total_combinations}")
                        print(f"Testing: n={n}, depth={d}, class_weight={cw}, max_features={mf}, min_samples_split={mss}")

                        start_time = time.time()

                        model, score, _, _ = randomForest(
                            X_array=X_array,
                            Y_encoded=Y_encoded,
                            groups=groups,
                            n_estimators=n,
                            max_depth=d,
                            class_weight=cw,
                            max_features=mf,
                            min_samples_split=mss
                        )

                        duration = time.time() - start_time

                        print(f"Score: {score} | Time: {duration:.2f}s")

                        run_result = {
                            "n_estimators": n,
                            "max_depth": d,
                            "class_weight": cw,
                            "max_features": mf,
                            "min_samples_split": mss,
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