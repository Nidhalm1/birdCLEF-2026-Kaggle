"""
paths.py — single place to configure paths for local vs Kaggle runs.

Switch ENV between "local" and "kaggle" to swap all paths at once.
All other scripts import from this file.
"""

import os

# ─────────────────────────────────────────────
#  ENV: "local"  or  "kaggle"
# ─────────────────────────────────────────────
ENV = "local"   # change to "kaggle" when running on Kaggle


if ENV == "kaggle":
    # ── Kaggle paths ─────────────────────────────────────────────────
    # Pre-saved .npy files (X, Y_encoded, groups, classes)
    SAVED_BASE_PATH = "/kaggle/input/datasets/emirhansagir/projmlsaved/saved/"

    # Where to write results, models, plots
    OUTPUT_BASE_PATH = "/kaggle/working/"

    # Raw audio data on Kaggle (adjust to actual mount path on the competition)
    CSV_PATH      = "/kaggle/input/birdclef-2026/train.csv"
    CSV2_PATH     = "/kaggle/input/birdclef-2026/train_soundscapes_labels.csv"
    AUDIO_PARENT  = "/kaggle/input/birdclef-2026/train_audio"
    AUDIO_PARENT2 = "/kaggle/input/birdclef-2026/train_soundscapes"

elif ENV == "local":
    # ── Local paths ──────────────────────────────────────────────────
    SAVED_BASE_PATH  = "saved/"
    OUTPUT_BASE_PATH = "outputs/"

    CSV_PATH      = os.path.join("csv", "train.csv")
    CSV2_PATH     = os.path.join("csv", "train_soundscapes_labels.csv")
    AUDIO_PARENT  = "train_audio"
    AUDIO_PARENT2 = "train_soundscapes"

else:
    raise ValueError(f"Unknown ENV: {ENV!r} (use 'local' or 'kaggle')")


# Convenience: print current config when imported
print(f"[paths.py] ENV={ENV}")
print(f"  SAVED_BASE_PATH  = {SAVED_BASE_PATH}")
print(f"  OUTPUT_BASE_PATH = {OUTPUT_BASE_PATH}")
