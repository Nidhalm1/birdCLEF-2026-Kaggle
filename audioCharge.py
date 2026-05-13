import os
import warnings
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed
from tqdm import tqdm

# Silence librosa's noisy warnings (empty frequency set, pysoundfile fallback, etc.)
warnings.filterwarnings("ignore", category=UserWarning,   module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

from paths import SAVED_BASE_PATH, SAVED_OUTPUT_PATH, CSV_PATH, AUDIO_PARENT


# ─────────────────────────────────────────────
#  FEATURE CONFIG
#  Set each flag to True/False to ablate.
#  This is the "full" config used for final runs.
# ─────────────────────────────────────────────
DEFAULT_FEATURE_CONFIG = {
    "mfcc":              True,   # 13 MFCCs → mean+std = 26 dims
    "mfcc_delta":        True,   # 1st derivative of MFCCs → 26 dims
    "mfcc_delta2":       True,   # 2nd derivative of MFCCs → 26 dims
    "chroma":            True,   # 12 chroma bins → 24 dims  [NEW]
    "mel":               True,   # 128-band mel spectrogram → 256 dims  [NEW]
    "spectral_centroid": True,   # 1 band → 2 dims
    "spectral_bandwidth":True,   # 1 band → 2 dims
    "spectral_rolloff":  True,   # 1 band → 2 dims
    "spectral_contrast": True,   # 7 bands → 14 dims  [NEW]
    "zcr":               True,   # zero-crossing rate → 2 dims
    "rms":               True,   # root-mean-square energy → 2 dims  [NEW]
    "tonnetz":           True,   # tonal centroid (6 dims) → 12 dims  [NEW]
}


def feature_dim(config=None):
    """Return the total feature vector length for a given config."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG
    dim = 0
    if config.get("mfcc"):              dim += 26
    if config.get("mfcc_delta"):        dim += 26
    if config.get("mfcc_delta2"):       dim += 26
    if config.get("chroma"):            dim += 24
    if config.get("mel"):               dim += 256
    if config.get("spectral_centroid"): dim += 2
    if config.get("spectral_bandwidth"):dim += 2
    if config.get("spectral_rolloff"):  dim += 2
    if config.get("spectral_contrast"): dim += 14
    if config.get("zcr"):               dim += 2
    if config.get("rms"):               dim += 2
    if config.get("tonnetz"):           dim += 12
    return dim


def extract_features(audio, sr, start_sec, duration=5, config=None):
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    start = int(start_sec * sr)
    end = int((start_sec + duration) * sr)
    segment = audio[start:end]
    if(len(segment)< duration*sr):
        segment = np.pad(segment, (0, duration * sr - len(segment))) #padding au cas ou 

    def stats(x):
        return np.concatenate([np.mean(x,axis=1), np.std(x, axis=1)])

    parts = []

    # MFCCs (original)
    if config.get("mfcc") or config.get("mfcc_delta") or config.get("mfcc_delta2"):
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        if config.get("mfcc"):
            parts.append(stats(mfcc))

         # Delta (original)
        if config.get("mfcc_delta"):
            delta = librosa.feature.delta(mfcc)
            parts.append(stats(delta))
        if config.get("mfcc_delta2"):
            delta2 = librosa.feature.delta(mfcc, order=2)
            parts.append(stats(delta2))

    # Chroma [NEW]
    #augmenter les donne noter l'ancien score et dire le nouveau avec ces ajout un par un et compnred la moyenne de delta c'est quoi delta ? 
    # comprendre le score auc
    # regarde chatgpt les etapes ajout deuxmeee model ect ? 
    # creer le git mnt 
    if config.get("chroma"):
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        parts.append(stats(chroma))

    # Mel spectrogram [NEW]
    if config.get("mel"):
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        parts.append(stats(mel_db))

     # Spectral features (original)
    if config.get("spectral_centroid"):
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        parts.append(stats(centroid))
    if config.get("spectral_bandwidth"):
        bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        parts.append(stats(bandwidth))
    if config.get("spectral_rolloff"):
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        parts.append(stats(rolloff))

    # Spectral contrast [NEW]
    if config.get("spectral_contrast"):
        contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
        parts.append(stats(contrast))

    # ZCR (original)
    if config.get("zcr"):
        zcr = librosa.feature.zero_crossing_rate(segment)
        parts.append(stats(zcr))

    # RMS energy [NEW]
    if config.get("rms"):
        rms = librosa.feature.rms(y=segment)
        parts.append(stats(rms))

    # Tonnetz [NEW]
    if config.get("tonnetz"):
        harmonic = librosa.effects.harmonic(segment)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        parts.append(stats(tonnetz))

    return np.concatenate(parts)


# ─────────────────────────────────────────────
#  PER-FILE WORKER (top-level so joblib can pickle it)
# ─────────────────────────────────────────────
def _process_one_file_principal(row, parent, config):
    """Process one row of train.csv. Returns (X_list, Y_list, groups_list) or None on failure."""
    full_path = os.path.join(parent, row["filename"])
    try:
        audio, sr = librosa.load(full_path, sr=32000)
    except Exception as e:
        return None, str(e)

    duration_ttl = len(audio) / sr
    X, Y, groups = [], [], []
    i = 0
    while i < duration_ttl:
        features = extract_features(audio=audio, sr=sr, start_sec=int(i), config=config)
        X.append(features)
        Y.append([row["primary_label"]])
        groups.append(row["filename"])
        i += 5
    return (X, Y, groups), None


def _process_one_file_grouped(filename, grouped, parent, config):
    """Process one (filename, group) from train_soundscapes_labels.csv."""
    full_path = os.path.join(parent, filename)
    try:
        audio, sr = librosa.load(full_path, sr=32000)
    except Exception as e:
        return None, str(e)

    X, Y, groups = [], [], []
    for _, row in grouped.iterrows():
        start_sec = int(row["start"].split(":")[-1])
        features = extract_features(audio=audio, sr=sr, start_sec=start_sec, config=config)
        labels = (row["primary_label"]).split(';')
        X.append(features)
        Y.append(labels)
        groups.append(filename)
    return (X, Y, groups), None


# ─────────────────────────────────────────────
#  DATASET BUILDERS — PARALLELIZED
# ─────────────────────────────────────────────
def build_dataset(csv, parent, maxIter=100, config=None, n_jobs=-1):
    """Build dataset from train_soundscapes_labels.csv (segmented files). Parallelized."""
    df = pd.read_csv(csv)
    grouped_iter = list(df.groupby("filename"))[:maxIter]
    print(f"parent: {parent}, csv: {csv}")
    print(f"Processing {len(grouped_iter)} files in parallel with n_jobs={n_jobs} ...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one_file_grouped)(fn, grp, parent, config)
        for fn, grp in tqdm(grouped_iter, desc="Extracting features", unit="file")
    )

    X, Y, groups = [], [], []
    n_loaded = n_failed = 0
    sample_errors = []
    for result, err in results:
        if result is None:
            n_failed += 1
            if len(sample_errors) < 5:
                sample_errors.append(err)
        else:
            n_loaded += 1
            X.extend(result[0]); Y.extend(result[1]); groups.extend(result[2])

    for e in sample_errors:
        print(f"Erreur exemple: {e}")
    print(f"Loaded {n_loaded} files OK, {n_failed} failed.")
    return X, Y, groups


def build_dataset_principal(csv, parent, maxIter=100, config=None, n_jobs=-1,
                             checkpoint_dir=None, batch_size=2000):
    """Build dataset from train.csv (continuous recordings, 5-sec windows).
    Parallelized with checkpointing: saves intermediate results every `batch_size`
    files so a crash near the end doesn't lose all progress.
    """
    df = pd.read_csv(csv)
    df = df.head(maxIter).reset_index(drop=True)
    print(f"parent: {parent}, csv: {csv}")
    print(f"Processing {len(df)} files in parallel with n_jobs={n_jobs} ...")
    print(f"Checkpoint every {batch_size} files → {checkpoint_dir or 'no checkpoints'}")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    X_all, Y_all, groups_all = [], [], []
    n_loaded = n_failed = 0
    sample_errors = []
    n_batches = (len(df) + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, len(df))
        batch_rows = [df.iloc[i] for i in range(start, end)]

        print(f"\n--- Batch {b+1}/{n_batches}  (rows {start}..{end-1}) ---")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_one_file_principal)(row, parent, config)
            for row in tqdm(batch_rows, desc=f"Batch {b+1}/{n_batches}", unit="file")
        )

        for result, err in results:
            if result is None:
                n_failed += 1
                if len(sample_errors) < 5:
                    sample_errors.append(err)
            else:
                n_loaded += 1
                X_all.extend(result[0]); Y_all.extend(result[1]); groups_all.extend(result[2])

        # ── Checkpoint: save partial X/Y/groups after each batch ────────
        if checkpoint_dir:
            np.save(os.path.join(checkpoint_dir, "X_partial.npy"),      np.array(X_all))
            np.save(os.path.join(checkpoint_dir, "Y_partial.npy"),      np.array(Y_all, dtype=object))
            np.save(os.path.join(checkpoint_dir, "groups_partial.npy"), np.array(groups_all))
            print(f"  Checkpoint saved: {len(X_all)} segments so far "
                  f"(loaded {n_loaded}, failed {n_failed})")

    for e in sample_errors:
        print(f"Erreur exemple: {e}")
    print(f"\nLoaded {n_loaded} files OK, {n_failed} failed.")
    return X_all, Y_all, groups_all


# ─────────────────────────────────────────────
#  BUILD & SAVE DATASET
#  Run this file directly to (re)build saved/X.npy etc.
#  Uses DEFAULT_FEATURE_CONFIG (= all features ON).
# ─────────────────────────────────────────────
def build_and_save(csv_path=None, parent=None, maxIter=200, save_path=None, config=None,
                    n_jobs=-1, batch_size=2000):
    if csv_path is None:
        csv_path = CSV_PATH
    if parent is None:
        parent = AUDIO_PARENT
    if save_path is None:
        # use SAVED_OUTPUT_PATH (writable) — not SAVED_BASE_PATH (may be read-only)
        save_path = SAVED_OUTPUT_PATH
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    print(f"Building dataset from {csv_path} ...")
    print(f"Feature dim (expected): {feature_dim(config)}")
    print(f"Writing outputs to: {save_path}")

    # Checkpoints go in save_path/checkpoints/
    checkpoint_dir = os.path.join(save_path, "checkpoints")

    X, Y, groups = build_dataset_principal(
        csv_path, parent=parent, maxIter=maxIter, config=config,
        n_jobs=n_jobs, checkpoint_dir=checkpoint_dir, batch_size=batch_size,
    )

    print(f"Built {len(X)} segments from {len(set(groups))} files")

    if len(X) == 0:
        print("ERROR: no audio loaded. Check ffmpeg / audio file paths.")
        return

    mlb = MultiLabelBinarizer()
    Y_encoded = mlb.fit_transform(Y)

    X_array = np.array(X)
    groups_array = np.array(groups)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "X.npy"),         X_array)
    np.save(os.path.join(save_path, "Y_encoded.npy"), Y_encoded)
    np.save(os.path.join(save_path, "groups.npy"),    groups_array)
    np.save(os.path.join(save_path, "classes.npy"),   mlb.classes_)

    print(f"Saved → X:{X_array.shape}  Y:{Y_encoded.shape}  groups:{groups_array.shape}")
    print(f"Num classes: {len(mlb.classes_)}")


if __name__ == "__main__":
    # n_jobs=-1  → use all logical cores
    # n_jobs=N   → use N workers (e.g. 4 on Kaggle)
    # batch_size: checkpoint every N files (so a crash near the end doesn't lose all work)
    # Edit maxIter as needed (number of audio files to process)
    build_and_save(maxIter=35550, n_jobs=-1, batch_size=2000)