"""
paths.py — single place to configure paths for local vs Kaggle runs.

Switch ENV between "local" and "kaggle" to swap all paths at once.
All other scripts import from this file.

Two separate "saved" paths:
  - SAVED_BASE_PATH   : where to READ pre-saved .npy files from (read-only on Kaggle)
  - SAVED_OUTPUT_PATH : where to WRITE new .npy files / checkpoints (writable)
On local they're the same folder. On Kaggle they differ.
"""

import os

# ─────────────────────────────────────────────
#  ENV: "local"  or  "kaggle"
# ─────────────────────────────────────────────
ENV = "local"   # change to "kaggle" when running on Kaggle


if ENV == "kaggle":
    # ── Kaggle paths ─────────────────────────────────────────────────
    # /kaggle/input/ is READ-ONLY.  /kaggle/working/ is writable.

    # Pre-saved .npy files (read only)
    SAVED_BASE_PATH   = "/kaggle/input/datasets/emirhansagir/projmlsaved/saved/"

    # Where to WRITE new .npy / checkpoints (writable)
    SAVED_OUTPUT_PATH = "/kaggle/working/saved/"

    # Where to write results, models, plots
    OUTPUT_BASE_PATH  = "/kaggle/working/"

    # Raw audio data on Kaggle (competition mount path)
    CSV_PATH      = "/kaggle/input/birdclef-2026/train.csv"
    CSV2_PATH     = "/kaggle/input/birdclef-2026/train_soundscapes_labels.csv"
    AUDIO_PARENT  = "/kaggle/input/birdclef-2026/train_audio"
    AUDIO_PARENT2 = "/kaggle/input/birdclef-2026/train_soundscapes"

elif ENV == "local":
    # ── Local paths ──────────────────────────────────────────────────
    SAVED_BASE_PATH   = "saved/"
    SAVED_OUTPUT_PATH = "saved/"     # local: read and write to same place
    OUTPUT_BASE_PATH  = "outputs/"

    CSV_PATH      = os.path.join("csv", "train.csv")
    CSV2_PATH     = os.path.join("csv", "train_soundscapes_labels.csv")
    AUDIO_PARENT  = "train_audio"
    AUDIO_PARENT2 = "train_soundscapes"

else:
    raise ValueError(f"Unknown ENV: {ENV!r} (use 'local' or 'kaggle')")


# Convenience: print current config when imported
print(f"[paths.py] ENV={ENV}")
print(f"  SAVED_BASE_PATH   = {SAVED_BASE_PATH}    (read)")
print(f"  SAVED_OUTPUT_PATH = {SAVED_OUTPUT_PATH}  (write)")
print(f"  OUTPUT_BASE_PATH  = {OUTPUT_BASE_PATH}")