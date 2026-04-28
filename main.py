from logisticRegression import logistic_regression_model_tests
from svm import svm_model_tests
from randomForestClassifier import randomForest_model_tests
from xgboostClassifier import xgboost_model, xboost_model_tests
from svm import train_model2
from audioCharge import *

import joblib
import os


OUTPUT_BASE_PATH = "/kaggle/working/"

def ensure_output_dir(rest_path):
    os.makedirs(OUTPUT_BASE_PATH + rest_path, exist_ok=True)


def test_logistic_regression(filename="logistic_regression_results.csv"):
    best_model, best_params, best_score, results = logistic_regression_model_tests()
    print("Best Logistic Regression Parameters:", best_params)
    print("Best Logistic Regression Score:", best_score)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    ensure_output_dir("results/")
    df.to_csv(OUTPUT_BASE_PATH + "results/" + filename, index=False)

    print(f"Results saved to {filename}")

    # Save model
    ensure_output_dir("models/")
    model_path = OUTPUT_BASE_PATH + "models/logistic_regression_best_model.pkl"
    joblib.dump(best_model, model_path)

    return best_model, best_params, best_score

def test_svm(filename="svm_results.csv"):
    best_model, best_params, best_score, results = svm_model_tests()
    print("Best SVM Parameters:", best_params)
    print("Best SVM Score:", best_score)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    ensure_output_dir("results/")
    df.to_csv(OUTPUT_BASE_PATH + "results/" + filename, index=False)

    print(f"Results saved to {filename}")

    # Save model
    ensure_output_dir("models/")
    model_path = OUTPUT_BASE_PATH + "models/svm_best_model.pkl"
    joblib.dump(best_model, model_path)

    return best_model, best_params, best_score


def test_xboost(filename="xgboost_results.csv"):
    best_model, best_params, best_score, results = xboost_model_tests()
    print("Best XGBoost Parameters:", best_params)
    print("Best XGBoost Score:", best_score)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    ensure_output_dir("results/")
    df.to_csv(OUTPUT_BASE_PATH + "results/" + filename, index=False)

    print(f"Results saved to {filename}")

    ensure_output_dir("models/")
    model_path = OUTPUT_BASE_PATH + "models/xgboost_best_model.pkl"
    joblib.dump(best_model, model_path)

    return best_model, best_params, best_score



def test_random_forest(filename="random_forest_results.csv"):
    best_model, best_params, best_score, results = randomForest_model_tests()
    print("Best RandomForest Parameters:", best_params)
    print("Best RandomForest Score:", best_score)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by score (best first)
    df = df.sort_values(by="auc_score", ascending=False)

    # Save CSV
    ensure_output_dir("results/")
    df.to_csv(OUTPUT_BASE_PATH + "results/" + filename, index=False)

    print(f"Results saved to {filename}")

    # Save model
    ensure_output_dir("models/")
    model_path = OUTPUT_BASE_PATH + "models/random_forest_best_model.pkl"
    joblib.dump(best_model, model_path)

    return best_model, best_params, best_score



def test_model(name):
    best_model, best_params, best_score, results = None, None, None, None
    if name == "logistic_regression":
        best_model, best_params, best_score = logistic_regression_model_tests()
    elif name == "svm":
        best_model, best_params, best_score = svm_model_tests()
    elif name == "xgboost":
        best_model, best_params, best_score = xboost_model_tests()
    elif name == "random_forest":
        best_model, best_params, best_score = randomForest_model_tests()
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
    ensure_output_dir("results/")
    result_file_path = f"{OUTPUT_BASE_PATH}results/{name}_results.csv"
    df.to_csv(result_file_path, index=False)

    print(f"Results saved to {result_file_path}")

    # Save model
    ensure_output_dir("models/")
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
    for model_name in ["logistic_regression", "svm", "xgboost", "random_forest"]:
        print(f"\n\n=== Testing {model_name.replace('_', ' ').title()} ===")
        test_model(model_name)