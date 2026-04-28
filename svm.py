from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time


def svm_model(X_array=None, Y_encoded=None, groups=None, C=1.0, gamma="scale"):

    print("\n/// SVM Model ///")
    print(f"Parameters: C={C}, gamma={gamma}")

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


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = OneVsRestClassifier(
        SVC(C=C, gamma=gamma, probability=True, max_iter=5000),
        n_jobs=-1
    )

    print("Training SVM model...")
    model.fit(X_train, Y_train)

    # Probabilities
    Y_pred_proba = model.predict_proba(X_test)

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


def svm_model_tests():
    C_values = [0.1, 1.0, 10.0]
    gamma_values = ["scale", 0.01, 0.1]

    total_combinations = len(C_values) * len(gamma_values)
    print("/// SVM Hyperparameter Tests ///")
    print(f"Testing combinations of: C={C_values}, gamma={gamma_values}")
    print(f"Total combinations to test: {total_combinations}")
    counter = 0

    results = []

    best_model = None
    best_score = 0
    best_params = {}

    print("Starting SVM hyperparameter tests...")
    print("loading data...")

    X_array = np.load("saved/X.npy", allow_pickle=True)
    Y_encoded = np.load("saved/Y_encoded.npy", allow_pickle=True)
    groups = np.load("saved/groups.npy", allow_pickle=True)

    for C in C_values:
        for gamma in gamma_values:
            counter += 1
            print("\n\n----------------------------------------")
            print(f"Testing combination {counter} of {total_combinations}")
            print(f"Testing: C={C}, gamma={gamma}")

            start_time = time.time()

            model, score, _, _ = svm_model(
                X_array=X_array,
                Y_encoded=Y_encoded,
                groups=groups,
                C=C,
                gamma=gamma
            )

            duration = time.time() - start_time

            print(f"Score: {score} | Time: {duration:.2f}s")

            run_result = {
                "C": C,
                "gamma": gamma,
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

def train_model2():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    import numpy as np

    
    X_array =np.load("X.npy",allow_pickle=True)
    Y_encoded = np.load("Y_encoded.npy", allow_pickle=True)
    groups = np.load("groups.npy",  allow_pickle=True)

    # Garder seulement les 26 premiers éléments de chaque feature
    X_array = np.array([x[:26] for x in X_array])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]
    

  
    model = OneVsRestClassifier(
        SVC(probability=True)
    )

    model.fit(X_train, Y_train)

    # Probabilités
    Y_pred_proba_list = model.predict_proba(X_test)

    Y_pred_proba = model.predict_proba(X_test)

    # AUC SAFE (comme avant)
    auc_scores = []

    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_pred_proba[:, i]

        if len(np.unique(y_true)) > 1:
            auc_scores.append(roc_auc_score(y_true, y_pred))

    auc_score = np.mean(auc_scores)

    print("AUC:", auc_score)
  

    return model, auc_score, X_test, Y_test
