import numpy as np
import pandas as pd
import os
import gc
import time
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from paths import OUTPUT_BASE_PATH

XGB_PARTIAL_CSV = os.path.join(OUTPUT_BASE_PATH, "results", "xgboost_partial.csv")


def xgboost_model(X_train, X_test, Y_train, Y_test, n_estimators=600, max_depth=6,
                  learning_rate=0.01, subsample=0.8, colsample_bytree=0.8):
    print("\n/// XGBoost Model ///")
    print(f"Parameters: n={n_estimators}, depth={max_depth}, lr={learning_rate}, "
          f"ss={subsample}, cs={colsample_bytree}")

    base_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",   # faster + lower memory than 'exact'
    )

    # MultiOutputClassifier with n_jobs=1 — fitting 206 binary classifiers in
    # parallel quintuples peak RAM. We let XGBoost itself use all cores.
    model = MultiOutputClassifier(base_model, n_jobs=1)

    print("Training XGBoost model...")
    model.fit(X_train, Y_train)

    Y_pred_proba_list = model.predict_proba(X_test)
    Y_pred_proba = np.stack([
        p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0])
        for p in Y_pred_proba_list
    ], axis=1).astype(np.float32)
    del Y_pred_proba_list

    auc_scores = []
    for i in range(Y_test.shape[1]):
        if len(np.unique(Y_test[:, i])) > 1:
            auc_scores.append(roc_auc_score(Y_test[:, i], Y_pred_proba[:, i]))
    auc_score = float(np.mean(auc_scores)) if auc_scores else 0.5

    del Y_pred_proba
    return model, auc_score, X_test, Y_test


def _save_partial(results):
    os.makedirs(os.path.dirname(XGB_PARTIAL_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(XGB_PARTIAL_CSV, index=False)


def _load_partial():
    if not os.path.exists(XGB_PARTIAL_CSV):
        return []
    try:
        return pd.read_csv(XGB_PARTIAL_CSV).to_dict("records")
    except Exception:
        return []


def xgboost_model_tests(X_train, X_test, Y_train, Y_test):
    # ─── Memory-safe XGB grid.
    # Old grid: 27 configs. Now 12.
    # Dropped: lr=0.1 (too aggressive, overfits with n=1000),
    #          n=1000 + depth=8 combo (heaviest).
    n_estimators = [300, 600]                # was [300, 600, 1000]
    max_depth = [4, 6]                       # was [4, 6, 8]
    learning_rate = [0.01, 0.05]             # was [0.01, 0.05, 0.1]
    subsample = [0.8]
    colsample_bytree = [0.8]

    total = (len(n_estimators) * len(max_depth) * len(learning_rate)
             * len(subsample) * len(colsample_bytree))
    print("/// XGBoost Hyperparameter Tests ///")
    print(f"Grid: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"learning_rate={learning_rate}, subsample={subsample}, "
          f"colsample_bytree={colsample_bytree}")
    print(f"Total combinations: {total}")
    print("Memory-safe grid (n=1000 + depth=8 dropped; tree_method='hist').")

    results = _load_partial()
    done_keys = {(r["n_estimators"], r["max_depth"], r["learning_rate"],
                  r["subsample"], r["colsample_bytree"]) for r in results}
    if done_keys:
        print(f"Resuming: {len(done_keys)} configs already done.")

    best_model = None
    best_score = max([r["auc_score"] for r in results], default=0)
    best_params = max(results, key=lambda r: r["auc_score"]) if results else {}

    counter = 0
    for n in n_estimators:
        for d in max_depth:
            for lr in learning_rate:
                for ss in subsample:
                    for cs in colsample_bytree:
                        counter += 1
                        key = (n, d, lr, ss, cs)
                        if key in done_keys:
                            print(f"\n[{counter}/{total}] Skip already-done: {key}")
                            continue

                        print("\n\n----------------------------------------")
                        print(f"Testing {counter}/{total}: n={n}, depth={d}, "
                              f"lr={lr}, ss={ss}, cs={cs}")

                        start_time = time.time()
                        try:
                            model, score, _, _ = xgboost_model(
                                X_train, X_test, Y_train, Y_test,
                                n_estimators=n, max_depth=d, learning_rate=lr,
                                subsample=ss, colsample_bytree=cs,
                            )
                        except MemoryError as e:
                            print(f"!!! MemoryError, recording NaN")
                            print(f"    {e}")
                            results.append({
                                "n_estimators": n, "max_depth": d, "learning_rate": lr,
                                "subsample": ss, "colsample_bytree": cs,
                                "auc_score": float("nan"),
                                "train_time_sec": time.time() - start_time,
                            })
                            _save_partial(results)
                            gc.collect()
                            continue

                        duration = time.time() - start_time
                        print(f"Score: {score:.4f} | Time: {duration:.1f}s")

                        run_result = {
                            "n_estimators": n, "max_depth": d, "learning_rate": lr,
                            "subsample": ss, "colsample_bytree": cs,
                            "auc_score": score, "train_time_sec": duration,
                        }
                        results.append(run_result)
                        _save_partial(results)

                        if score > best_score:
                            best_score = score
                            best_params = run_result
                            best_model = model
                        else:
                            del model
                        
                        gc.collect()
                        print(f"Current best: {best_score:.4f} @ {best_params}")

    print(f"\nBest score: {best_score}")
    print(f"Best params: {best_params}")
    return best_model, best_params, best_score, results
