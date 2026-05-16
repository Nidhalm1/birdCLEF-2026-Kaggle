"""
train_vs_test_auc.py  –  Compute train AUC for saved best models
=================================================================
Loads saved .pkl models and the saved X/Y arrays, re-evaluates on
BOTH train and test splits, and prints a clean comparison table.

NO RETRAINING — uses joblib-saved models.

Usage:
    python train_vs_test_auc.py

Requirements: same environment as main.py (sklearn, xgboost, numpy, joblib).
Outputs:
    outputs/results/train_vs_test_auc.csv
    (printed table to stdout)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
# Edit these to match your local layout
SAVED_PATH   = "saved/"          # where X.npy, Y_encoded.npy, groups.npy live
MODELS_PATH  = "outputs/models/" # where the .pkl files live
RESULTS_PATH = "outputs/results/"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_best_model.pkl",
    "SVM":                 "svm_best_model.pkl",
    "Random Forest":       "random_forest_best_model.pkl",
    "XGBoost":             "xgboost_best_model.pkl",
}

# SVM was trained on a 30k subsample — flag it
SVM_SUBSAMPLE = 30_000
RANDOM_STATE  = 42


def mean_auc(model, X, Y):
    """Compute mean per-class ROC-AUC.
    Handles both:
      - MultiOutputClassifier: predict_proba returns a LIST of (n, 2) arrays
      - OneVsRestClassifier:   predict_proba returns a single (n, K) array
    """
    proba = model.predict_proba(X)

    if isinstance(proba, list):
        # MultiOutputClassifier — one array per class
        Y_pred = np.stack([
            p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.zeros(X.shape[0])
            for p in proba
        ], axis=1)
    else:
        # OneVsRestClassifier — single (n_samples, n_classes) array
        Y_pred = proba

    aucs = []
    for i in range(Y.shape[1]):
        if len(np.unique(Y[:, i])) > 1:
            aucs.append(roc_auc_score(Y[:, i], Y_pred[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading saved arrays …")
    X_array  = np.load(os.path.join(SAVED_PATH, "X.npy"),         allow_pickle=True)
    Y_enc    = np.load(os.path.join(SAVED_PATH, "Y_encoded.npy"), allow_pickle=True)
    groups   = np.load(os.path.join(SAVED_PATH, "groups.npy"),    allow_pickle=True)
    print(f"  X: {X_array.shape}  Y: {Y_enc.shape}  groups: {groups.shape}")

    # ── Reproduce the exact same split ────────────────────────────────────────
    print("Reproducing train/test split (GroupShuffleSplit seed=42) …")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_array, Y_enc, groups))

    X_train_raw, X_test_raw = X_array[train_idx], X_array[test_idx]
    Y_train,     Y_test     = Y_enc[train_idx],   Y_enc[test_idx]
    print(f"  Train: {X_train_raw.shape[0]} samples  |  Test: {X_test_raw.shape[0]} samples")

    results = []

    for model_name, fname in MODEL_FILES.items():
        model_path = os.path.join(MODELS_PATH, fname)
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_name}: {model_path} not found.")
            results.append({"model": model_name, "train_auc": None, "test_auc": None, "gap": None})
            continue

        print(f"\n── {model_name} ──────────────────────────────────────────")
        model = joblib.load(model_path)
        print(f"  Loaded: {model_path}")

        # Scaling — LR and SVM use StandardScaler inside their training scripts.
        # We must replicate the same scaling here (fit on train only).
        # RF and XGBoost do NOT use a scaler.
        needs_scaling = model_name in ("Logistic Regression", "SVM")
        if needs_scaling:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_raw)
            X_te = scaler.transform(X_test_raw)
        else:
            X_tr = X_train_raw
            X_te = X_test_raw

        # SVM was trained on a subsample — reproduce the same subsample for train AUC
        if model_name == "SVM":
            rng = np.random.RandomState(RANDOM_STATE)
            sub_idx = rng.choice(X_tr.shape[0], size=min(SVM_SUBSAMPLE, X_tr.shape[0]), replace=False)
            X_tr_eval = X_tr[sub_idx]
            Y_tr_eval = Y_train[sub_idx]
            print(f"  SVM: evaluating train AUC on same {SVM_SUBSAMPLE}-sample subsample used for training.")
        else:
            X_tr_eval = X_tr
            Y_tr_eval = Y_train

        print("  Computing train AUC …", flush=True)
        train_auc = mean_auc(model, X_tr_eval, Y_tr_eval)
        print(f"  Train AUC: {train_auc:.4f}")

        print("  Computing test AUC …", flush=True)
        test_auc = mean_auc(model, X_te, Y_test)
        print(f"  Test AUC:  {test_auc:.4f}")

        gap = test_auc - train_auc
        print(f"  Gap (test - train): {gap:+.4f}")

        results.append({
            "model":     model_name,
            "train_auc": round(train_auc, 4),
            "test_auc":  round(test_auc,  4),
            "gap":       round(gap, 4),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("TRAIN vs TEST AUC SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(RESULTS_PATH, "train_vs_test_auc.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # ── LaTeX snippet for the report ─────────────────────────────────────────
    print("\n─── LaTeX table snippet (paste into report) ───\n")
    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lcccc}")
    print(r"  \toprule")
    print(r"  \textbf{Model} & \textbf{Train AUC} & \textbf{Test AUC} & \textbf{Gap} & \textbf{Regime} \\")
    print(r"  \midrule")
    for r in results:
        if r["train_auc"] is None:
            regime = "N/A"
            line = f"  {r['model']} & -- & -- & -- & {regime} \\\\"
        else:
            g = r["gap"]
            if g > -0.02:
                regime = "Well regularised"
            elif g > -0.08:
                regime = "Slight over-fit"
            else:
                regime = "Over-fit"
            line = f"  {r['model']} & {r['train_auc']:.4f} & {r['test_auc']:.4f} & {g:+.4f} & {regime} \\\\"
        print(line)
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Train vs.\ test mean ROC-AUC. Gap = test $-$ train (negative = over-fit).}")
    print(r"  \label{tab:train-test}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()