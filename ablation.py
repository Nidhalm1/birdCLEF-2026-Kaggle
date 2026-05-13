"""
ablation.py  –  Feature engineering ablation study
====================================================
Runs a fast model (Logistic Regression, best params from grid search)
on every feature-group combination and saves a CSV so you can build
a table for the report.

Usage:
    python ablation.py

Output:
    outputs/ablation/ablation_results.csv
    outputs/ablation/ablation_plot.png
    outputs/logs/ablation_<timestamp>.log
"""

from logger import setup_logging, log_section, close_logging
setup_logging("ablation")

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

from audioCharge import (
    build_dataset_principal,
    DEFAULT_FEATURE_CONFIG,
    feature_dim,
)
from paths import OUTPUT_BASE_PATH, CSV_PATH, AUDIO_PARENT

OUTPUT_PATH     = os.path.join(OUTPUT_BASE_PATH, "ablation/")
MAX_FILES       = 5000        # increase for a real run
LR_PARAMS       = {"C": 0.01, "max_iter": 10000}   # best from your grid search


# ─────────────────────────────────────────────
#  FEATURE GROUPS TO ABLATE
#  Each entry: (group_name, list_of_keys_to_enable)
#  Progressive: each adds one group on top of the previous
# ─────────────────────────────────────────────
FEATURE_GROUPS = [
    ("mfcc_only",           ["mfcc"]),
    ("mfcc+deltas",         ["mfcc", "mfcc_delta", "mfcc_delta2"]),
    ("mfcc+deltas+spectral",["mfcc", "mfcc_delta", "mfcc_delta2",
                              "spectral_centroid", "spectral_bandwidth",
                              "spectral_rolloff", "zcr"]),
    ("+ chroma",            ["mfcc", "mfcc_delta", "mfcc_delta2",
                              "spectral_centroid", "spectral_bandwidth",
                              "spectral_rolloff", "zcr", "chroma"]),
    ("+ spectral_contrast", ["mfcc", "mfcc_delta", "mfcc_delta2",
                              "spectral_centroid", "spectral_bandwidth",
                              "spectral_rolloff", "zcr", "chroma",
                              "spectral_contrast"]),
    ("+ rms",               ["mfcc", "mfcc_delta", "mfcc_delta2",
                              "spectral_centroid", "spectral_bandwidth",
                              "spectral_rolloff", "zcr", "chroma",
                              "spectral_contrast", "rms"]),
    ("+ tonnetz",           ["mfcc", "mfcc_delta", "mfcc_delta2",
                              "spectral_centroid", "spectral_bandwidth",
                              "spectral_rolloff", "zcr", "chroma",
                              "spectral_contrast", "rms", "tonnetz"]),
    ("all_features (+ mel)",list(DEFAULT_FEATURE_CONFIG.keys())),
]


def make_config(enabled_keys):
    return {k: (k in enabled_keys) for k in DEFAULT_FEATURE_CONFIG}


def build_and_split(config):
    """Build dataset with given feature config, encode, return train/test splits."""
    print("loading data...")
    X_list, Y_list, groups_list = build_dataset_principal(
        CSV_PATH, AUDIO_PARENT, maxIter=MAX_FILES, config=config
    )

    mlb = MultiLabelBinarizer()
    Y_encoded = mlb.fit_transform(Y_list)
    X_array   = np.array(X_list)
    groups    = np.array(groups_list)

    print(f"X shape: {X_array.shape}  Y shape: {Y_encoded.shape}")
    print(f"Unique groups (files): {len(set(groups))}")
    print(f"Unique classes: {Y_encoded.shape[1]}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    print(f"Train samples: {len(train_idx)}  Test samples: {len(test_idx)}")
    print(f"Train groups: {len(set(groups[train_idx]))}  Test groups: {len(set(groups[test_idx]))}")

    return (
        X_array[train_idx], X_array[test_idx],
        Y_encoded[train_idx], Y_encoded[test_idx],
    )


def evaluate(X_train, X_test, Y_train, Y_test):
    """Train LR with best params and return mean AUC-ROC."""
    valid = [i for i in range(Y_train.shape[1]) if len(np.unique(Y_train[:, i])) > 1]
    print(f"Evaluable classes (≥2 unique values in train): {len(valid)}/{Y_train.shape[1]}")
    if not valid:
        return 0.5

    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train)
    Xte     = scaler.transform(X_test)

    model = MultiOutputClassifier(LogisticRegression(**LR_PARAMS), n_jobs=-1)
    model.fit(Xtr, Y_train[:, valid])

    proba_list = model.predict_proba(Xte)
    Y_pred = np.stack([
        p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0])
        for p in proba_list
    ], axis=1)

    aucs = []
    for i, vi in enumerate(valid):
        if len(np.unique(Y_test[:, vi])) > 1:
            aucs.append(roc_auc_score(Y_test[:, vi], Y_pred[:, i]))

    print(f"Classes scored on (≥2 unique values in BOTH train and test): {len(aucs)}")

    return float(np.mean(aucs)) if aucs else 0.5


def run_ablation():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    results = []

    for group_name, enabled_keys in FEATURE_GROUPS:
        config = make_config(enabled_keys)
        dim    = feature_dim(config)
        log_section(f"Config: {group_name}  |  dim={dim}")
        print(f"Features: {enabled_keys}")

        t0 = time.time()
        try:
            X_train, X_test, Y_train, Y_test = build_and_split(config)
            auc = evaluate(X_train, X_test, Y_train, Y_test)
        except Exception as e:
            print(f"ERROR: {e}")
            auc = None

        duration = time.time() - t0
        print(f"Score: {auc} | Time: {duration:.2f}s")

        results.append({
            "config":      group_name,
            "features":    ", ".join(enabled_keys),
            "feature_dim": dim,
            "auc_score":   auc,
            "time_sec":    round(duration, 2),
        })

    df = pd.DataFrame(results)
    csv_out = os.path.join(OUTPUT_PATH, "ablation_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nResults saved to {csv_out}")

    # ── Plot ──────────────────────────────────────────────────────────
    df_plot = df.dropna(subset=["auc_score"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_plot["config"], df_plot["auc_score"], color="steelblue")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Mean AUC-ROC")
    ax.set_title("Feature engineering ablation (Logistic Regression, best params)")
    ax.set_xlim(0.5, 1.0)
    plt.tight_layout()
    plot_out = os.path.join(OUTPUT_PATH, "ablation_plot.png")
    fig.savefig(plot_out, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_out}")

    log_section("ABLATION SUMMARY")
    print(df[["config", "feature_dim", "auc_score"]].to_string(index=False))

    return df


if __name__ == "__main__":
    try:
        run_ablation()
    finally:
        close_logging()
