from logisticRegression import logistic_regression_model_tests
from svm import svm_model_tests
from randomForestClassifier import randomForest_model_tests
from xgboostClassifier import xgboost_model, xboost_model_tests
from audioCharge import *

import joblib
import time
import os


OUTPUT_BASE_PATH = "/kaggle/working/"

def ensure_output_dir(rest_path):
    os.makedirs(OUTPUT_BASE_PATH + rest_path, exist_ok=True)


def test_model(name):
    best_model, best_params, best_score, results = None, None, None, None
    if name == "logistic_regression":
        best_model, best_params, best_score = logistic_regression_model_tests()
    elif name == "svm":
        best_model, best_params, best_score = svm_model_tests()
    elif name == "random_forest":
        best_model, best_params, best_score = randomForest_model_tests()
    elif name == "xgboost":
        best_model, best_params, best_score = xboost_model_tests()
    else:
        print(f"Unknown model name: {name}")
        return None, None, None
    
    print(f"Best {name.capitalize()} Parameters:", best_params)
    print(f"Best {name.capitalize()} Score:", best_score)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    ensure_output_dir(f"{OUTPUT_BASE_PATH}results/")
    result_file_path = f"{OUTPUT_BASE_PATH}results/{name}_results.csv"
    df.to_csv(result_file_path, index=False)

    print(f"Results saved to {result_file_path}")

    # Save model
    ensure_output_dir(f"{OUTPUT_BASE_PATH}models/")
    model_path = f"{OUTPUT_BASE_PATH}models/{name}_best_model.pkl"
    joblib.dump(best_model, model_path)

    return best_model, best_params, best_score



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
    start = time.time()
    test_start = time.time()
    test_end = time.time()
    times = []
    model_names = ["svm", "random_forest", "xgboost"] # ["logistic_regression", "svm", "random_forest", "xgboost"]

    for model_name in model_names:
        print(f"\n\n=== Testing {model_name.replace('_', ' ').title()} ===")
        test_start = time.time()
        test_model(model_name)
        test_end = time.time()
        model_time = test_end - test_start
        print(f"Time taken for {model_name}: {model_time:.2f}s")
        times.append(model_time)
    
    end = time.time()
    print(f"Total time taken: {end - start:.2f}s")
    print("Individual model times:")
    for model_name, model_time in zip(model_names, times):
        print(f"{model_name.replace('_', ' ').title()}: {model_time:.2f}s")