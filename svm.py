from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# ─────────────────────────────────────────────
#  SVM SUBSAMPLE SIZE
#  SVM training cost is O(n^2)..O(n^3) in samples and O(n^2) in memory.
#  Full dataset (212k train samples) would need ~360 GB RAM for the
#  kernel matrix. We subsample the training set down to SVM_TRAIN_SUBSAMPLE
#  rows (stratified-ish via random selection within groups is overkill here;
#  uniform random is fine and standard practice).
# ─────────────────────────────────────────────
SVM_TRAIN_SUBSAMPLE = 30000   # ~7 GB kernel matrix, ~5-15 min per fit


def svm_model(X_train, X_test, Y_train, Y_test, C=1.0, gamma="scale", max_iter=1000,
              subsample=SVM_TRAIN_SUBSAMPLE, random_state=42):

    print("\n/// SVM Model ///")
    print(f"Parameters: C={C}, gamma={gamma}, max_iter={max_iter}")

    # ── Subsample training set if too large ────────────────────────────
    if subsample is not None and X_train.shape[0] > subsample:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X_train.shape[0], size=subsample, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        print(f"Subsampled training set: {X_train.shape[0]} samples (was larger)")
    else:
        print(f"Training set size: {X_train.shape[0]} samples")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = OneVsRestClassifier(
        SVC(C=C, gamma=gamma, probability=True, max_iter=max_iter),
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


def svm_model_tests(X_train, X_test, Y_train, Y_test):
    # ─── Reduced grid: SVM is slow even on subsampled data ────────────
    C_values = [0.1, 1.0, 10.0]                # was [0.001, 0.01, 0.1, 1.0, 10.0]
    gamma_values = ["scale", 0.01]              # was ["scale", 0.01, 0.1]
    max_iter_values = [1000, 5000]              # was [500, 1000, 5000, 10000]

    total_combinations = len(C_values) * len(gamma_values) * len(max_iter_values)
    print("/// SVM Hyperparameter Tests ///")
    print(f"Testing combinations of: C={C_values}, gamma={gamma_values}, max_iter={max_iter_values}")
    print(f"Total combinations to test: {total_combinations}")
    print(f"NOTE: SVM uses a subsample of {SVM_TRAIN_SUBSAMPLE} training samples (full would OOM).")
    counter = 0

    results = []

    best_model = None
    best_score = 0
    best_params = {}

    print("Starting SVM hyperparameter tests...")

    for C in C_values:
        for gamma in gamma_values:
            for max_iter in max_iter_values:
                counter += 1
                print("\n\n----------------------------------------")
                print(f"Testing combination {counter} of {total_combinations}")
                print(f"Testing: C={C}, gamma={gamma}, max_iter={max_iter}")

                start_time = time.time()

                model, score, _, _ = svm_model(
                    X_train, X_test, Y_train, Y_test,
                    C=C,
                    gamma=gamma,
                    max_iter=max_iter
                )

                duration = time.time() - start_time

                print(f"Score: {score} | Time: {duration:.2f}s")

                run_result = {
                    "C": C,
                    "gamma": gamma,
                    "max_iter": max_iter,
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
