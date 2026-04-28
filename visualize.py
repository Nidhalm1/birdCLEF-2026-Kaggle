import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(input_folder="outputs/results/", output_folder="outputs/plots/"):
    os.makedirs(output_folder, exist_ok=True)

    model_scores = {}

    for file in os.listdir(input_folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(input_folder, file)
        model_name = file.replace(".csv", "").replace("_results", "")

        print(f"Processing: {file}")

        df = pd.read_csv(file_path)

        if "auc_score" not in df.columns:
            print(f"Skipping {file} (no auc_score column)")
            continue

        possible_cols = [c for c in df.columns if c not in ["auc_score", "train_time_sec"]]
        if len(possible_cols) == 0:
            print(f"Skipping {file} (no valid x column)")
            continue

        # Create model-specific folder
        model_output_dir = os.path.join(output_folder, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for col in possible_cols:

            df_best = (
                df.groupby(col, as_index=False)["auc_score"]
                .max()
                .sort_values(by=col)
            )

            # Plot
            plt.figure()
            plt.plot(df_best[col], df_best["auc_score"], marker='o')
            plt.xlabel(col)
            plt.ylabel("Best AUC Score")
            plt.title(f"{model_name} - Best AUC vs {col}")

            if df_best[col].dtype != object:
                plt.xscale("log")

            plot_path = os.path.join(model_output_dir, f"{model_name}_{col}_vs_auc.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"Saved plot: {plot_path}")

        # Store best overall score
        best_score = df["auc_score"].max()
        model_scores[model_name] = best_score

    # Final comparison plot
    if len(model_scores) > 0:
        plt.figure()
        names = list(model_scores.keys())
        scores = list(model_scores.values())

        plt.bar(names, scores)
        plt.ylabel("Best AUC Score")
        plt.title("Model Comparison")

        comparison_path = os.path.join(output_folder, "model_comparison.png")
        plt.savefig(comparison_path)
        plt.close()

        print(f"Saved comparison plot: {comparison_path}")
    else:
        print("No valid CSV files found.")


if __name__ == "__main__":
    input_folder = "outputs/results/"
    output_folder = "outputs/plots/"
    
    plot_results(input_folder, output_folder)