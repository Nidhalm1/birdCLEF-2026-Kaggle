from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def train_model2():
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import numpy as np

    
    X_array =np.load("X.npy",allow_pickle=True)
    Y_encoded = np.load("Y_encoded.npy", allow_pickle=True)
    groups = np.load("groups.npy",  allow_pickle=True)

    # Garder seulement les 26 premiers éléments de chaque feature
    X_array = np.array([x[:26] for x in X_array])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_array, Y_encoded, groups))

    X_train, X_test = X_array[train_idx], X_array[test_idx]
    Y_train, Y_test = Y_encoded[train_idx], Y_encoded[test_idx]
    

  
    model = OneVsRestClassifier(
        SVC(probability=True)
    )

    model.fit(X_train, Y_train)

    # Probabilités
    Y_pred_proba_list = model.predict_proba(X_test)

    Y_pred_proba = model.predict_proba(X_test)

    # AUC SAFE (comme avant)
    auc_scores = []

    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_pred_proba[:, i]

        if len(np.unique(y_true)) > 1:
            auc_scores.append(roc_auc_score(y_true, y_pred))

    auc_score = np.mean(auc_scores)

    print("AUC:", auc_score)
  

    return model, auc_score, X_test, Y_test
