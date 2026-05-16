from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import time
import gc
import os

from paths import OUTPUT_BASE_PATH

RF_PARTIAL_CSV = os.path.join(OUTPUT_BASE_PATH, "results", "random_forest_partial.csv")


def randomForest(X_train, X_test, Y_train, Y_test, n_estimators=600, max_depth=10,
                 class_weight="balanced", max_features="sqrt", min_samples_split=2):
    print("\n/// RandomForest Model ///")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"class_weight={class_weight}, max_features={max_features}, "
          f"min_samples_split={min_samples_split}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        max_features=max_features,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, Y_train)

    # predict_proba returns a list of arrays. Stack the positive-class column.
    # We free the intermediate list as we go to keep memory down.
    Y_pred_proba_list = model.predict_proba(X_test)
    Y_pred_proba = np.stack([
        p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0])
        for p in Y_pred_proba_list
    ], axis=1).astype(np.float32)  # float32 halves memory
    del Y_pred_proba_list

    # AUC
    auc_scores = []
    for i in range(Y_test.shape[1]):
        if len(np.unique(Y_test[:, i])) > 1:
            auc_scores.append(roc_auc_score(Y_test[:, i], Y_pred_proba[:, i]))
    auc_score = float(np.mean(auc_scores)) if auc_scores else 0.5

    del Y_pred_proba
    return model, auc_score, X_test, Y_test


def _save_partial(results):
    os.makedirs(os.path.dirname(RF_PARTIAL_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(RF_PARTIAL_CSV, index=False)


def _load_partial():
    """Return list of dicts that have already been tested (resume support)."""
    if not os.path.exists(RF_PARTIAL_CSV):
        return []
    try:
        df = pd.read_csv(RF_PARTIAL_CSV)
        return df.to_dict("records")
    except Exception:
        return []


def randomForest_model_tests(X_train, X_test, Y_train, Y_test):
    # ─── Memory-safe RF grid for full 265k dataset.
    # Crashes were happening with n=1000, max_depth=None (unlimited).
    # Lessons from the LR/SVM run:
    #   - LR with C=0.01 already gives 0.867 — RF needs to beat that to matter
    #   - Trees with max_depth=None blow up: 1000 trees × ~50k nodes each × 206 classes
    #     ≈ very high RAM
    # We REMOVE max_depth=None and DROP n=1000 to stay under ~12 GB.
    # 2*2*1*2*3 = 24 configs (down from 54)
    n_estimators = [300, 600]                    # was [300, 600, 1000]
    max_depth = [10, 20]                         # was [10, 20, None] — None caused OOM
    class_weight = ["balanced"]
    max_features = ["sqrt", "log2"]
    min_samples_split = [2, 5, 10]

    total = len(n_estimators) * len(max_depth) * len(class_weight) * len(max_features) * len(min_samples_split)
    print("/// RandomForest Hyperparameter Tests ///")
    print(f"Grid: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"class_weight={class_weight}, max_features={max_features}, "
          f"min_samples_split={min_samples_split}")
    print(f"Total combinations: {total}")
    print("Memory-safe grid: dropped max_depth=None and n_estimators=1000.")

    # ── Resume support ─────────────────────────────────────────────────
    results = _load_partial()
    done_keys = {(r["n_estimators"], r["max_depth"], r["max_features"], r["min_samples_split"])
                 for r in results}
    if done_keys:
        print(f"Resuming: {len(done_keys)} configs already completed (loaded from {RF_PARTIAL_CSV}).")

    best_model = None
    best_score = max([r["auc_score"] for r in results], default=0)
    best_params = max(results, key=lambda r: r["auc_score"]) if results else {}

    counter = 0
    for n in n_estimators:
        for d in max_depth:
            for cw in class_weight:
                for mf in max_features:
                    for mss in min_samples_split:
                        counter += 1
                        key = (n, d, mf, mss)
                        if key in done_keys:
                            print(f"\n[{counter}/{total}] Skipping already-done config: {key}")
                            continue

                        print("\n\n----------------------------------------")
                        print(f"Testing combination {counter} of {total}")
                        print(f"n={n}, depth={d}, cw={cw}, mf={mf}, mss={mss}")

                        start_time = time.time()
                        try:
                            model, score, _, _ = randomForest(
                                X_train, X_test, Y_train, Y_test,
                                n_estimators=n, max_depth=d, class_weight=cw,
                                max_features=mf, min_samples_split=mss,
                            )
                        except MemoryError as e:
                            print(f"!!! MemoryError on this config — recording NaN, continuing")
                            print(f"    {e}")
                            results.append({
                                "n_estimators": n, "max_depth": d, "class_weight": cw,
                                "max_features": mf, "min_samples_split": mss,
                                "auc_score": float("nan"),
                                "train_time_sec": time.time() - start_time,
                            })
                            _save_partial(results)
                            gc.collect()
                            continue

                        duration = time.time() - start_time
                        print(f"Score: {score:.4f} | Time: {duration:.1f}s")

                        run_result = {
                            "n_estimators": n, "max_depth": d, "class_weight": cw,
                            "max_features": mf, "min_samples_split": mss,
                            "auc_score": score, "train_time_sec": duration,
                        }
                        results.append(run_result)
                        _save_partial(results)   # checkpoint after every config

                        if score > best_score:
                            best_score = score
                            best_params = run_result
                            best_model = model
                        else:
                            del model  # free memory if not the new best

                        gc.collect()
                        print(f"Current best: {best_score:.4f} @ {best_params}")

    print(f"\nBest score: {best_score}")
    print(f"Best params: {best_params}")

    return best_model, best_params, best_score, results
