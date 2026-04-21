from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np


def randomForest():

    X_array =np.load("saved/X.npy",allow_pickle=True)
    Y_encoded = np.load("saved/Y_encoded.npy", allow_pickle=True)
    groups = np.load("saved/groups.npy",  allow_pickle=True)

    # Garder seulement les 26 premiers éléments de chaque feature
    #X_array = np.array([x[] for x in X_array])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]
    

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, Y_train)

    # Probabilités
    Y_pred_proba_list = model.predict_proba(X_test)

    Y_pred_proba = np.stack([
        proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0])
        for proba in Y_pred_proba_list
    ], axis=1)

    # AUC SAFE
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
    #print("Y_encoded") [[1 0 0 ... 0 0 0][0 1 0 ... 0 0 0]]
    #print(Y_pred_proba) [[0.1322108  0.01       0.0295151  ... 0.005      0.         0.015     ][0.16297093 0.02       0.00693374 ... 0.02       0.01       0.015     ]
    print("Y_pred_proba_list")
    print(Y_pred_proba_list) # array([[1.        , 0.        ],[1.        , 0.        ],[0.995     , 0.005     ],...,[0.98058876, 0.01941124]], shape=(872, 2)), array([[0.995     , 0.005     ],
    # chaque array est une espace et et à lintereieur c'est des segment y en 872


    return model, auc_score, X_test, Y_test
