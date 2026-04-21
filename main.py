from randomForestClassifier import randomForest
from xgboostClassifier import xgboost_model, xboost_model_tests
from svm import train_model2
from audioCharge import *

import numpy as np
import os

def test_xboost(filename="xgboost_results.csv"):
    best_params, best_score, results = xboost_model_tests()
    print("Best XGBoost Parameters:", best_params)
    print("Best XGBoost Score:", best_score)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    df.to_csv(filename, index=False)

    print(f"Results saved to {filename}")

    return best_params, best_score

def main():
    csv_path = os.path.join("csv", "train.csv")
    csv2_path = os.path.join("csv", "train_soundscapes_labels.csv")
    parent1 = "train_audio"
    parent2 = "train_soundscapes"

    #X1, Y1 , groups1 = build_dataset_principal(csv_path, parent=parent1, maxIter=50)
    #X2, Y2 , groups2 = build_dataset(csv2_path, parent=parent2, maxIter=50)
    #X1.extend(X2)
    #Y1.extend(Y2)
    #groups1.extend(groups2)


    model, score, X_test, y_test = xgboost_model()

    print("Score :", score)


if __name__ == "__main__":
    test_xboost()