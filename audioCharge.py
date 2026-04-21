import os
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit # NOUVEAU : L'import

from sklearn.model_selection import train_test_split


def extract_features(audio, sr, start_sec, duration=5):
    start = int(start_sec * sr)
    end = int((start_sec + duration) * sr)
    segment = audio[start:end]
    if(len(segment)< duration*sr):
        segment = np.pad(segment, (0, duration * sr - len(segment))) #padding au cas ou 

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

     # Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

     # Spectral features
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(segment)

    #augmenter les donne noter l'ancien score et dire le nouveau avec ces ajout un par un et compnred la moyenne de delta c'est quoi delta ? 
    # comprendre le score auc
    # regarde chatgpt les etapes ajout deuxmeee model ect ? 
    # creer le git mnt 
    def stats(x):
        return np.concatenate([np.mean(x,axis=1), np.std(x, axis=1)])
    features = np.concatenate([
        stats(mfcc),
        stats(delta),
        stats(delta2),
        stats(centroid),
        stats(bandwidth),
        stats(rolloff),
        stats(zcr)
    ])

    return features

    


def build_dataset(csv, parent , maxIter= 100):
    df = pd.read_csv(csv)
    res = df.groupby("filename")
    X = []
    Y = []
    groups = []  # NOUVEAU : On crée une liste pour stocker le nom du fichier source
    count = 0
    for filename, grouped in res:
        if count >=maxIter:
            break
        full_path = os.path.join(parent,filename)
        try:
            audio, sr = librosa.load(full_path, sr=32000)
        except:
            continue
        for _ , row in grouped.iterrows():
            start_sec = int(row['start'].split(":")[-1])            
            features = extract_features(audio= audio, sr = sr,start_sec=start_sec)
            labels = (row["primary_label"]).split(';')
            X.append(features)
            Y.append(labels)
            groups.append(filename)
        count+=1
    return X,Y, groups


def build_dataset_principal(csv, parent , maxIter= 100):
    df = pd.read_csv(csv)
    print(f"parent: {parent}, csv: {csv}")
    X = []
    Y = []
    groups = []  # NOUVEAU : On crée une liste pour stocker le nom du fichier source
    for _ , row in df.iterrows():
        full_path = os.path.join(parent,row["filename"])
        try:
            audio, sr = librosa.load(full_path, sr=32000)
        except Exception as e:
            #print(f"Erreur fichier {full_path}: {e}")
            continue
        duration_ttl = len(audio)/sr
        i = 0
        while i < duration_ttl:
            features = extract_features(audio= audio, sr =sr,start_sec=int(i))
            X.append(features)
            Y.append([row["primary_label"]])
            groups.append(row["filename"])
            i+=5
        maxIter-=1
        if maxIter == 0 : 
            break

    return X, Y , groups



