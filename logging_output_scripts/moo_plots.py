from logging_output_scripts.utils import get_csv_df, get_normalized_df, check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from utils import datasets_map
from sklearn.preprocessing import MinMaxScaler

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"
hypervolume = "metrics.hypervolume"
spread = "metrics.spread"


def create_plots():
    """
    Uses seaborn-package to create violin-Plots comparing model performances
    on multiple datasets
    """
    sns.set_style("whitegrid")
    sns.set_theme(style="whitegrid",
                  font="Times New Roman",
                  font_scale=1.7,
                  rc={
                      "lines.linewidth": 1,
                      "pdf.fonttype": 42,
                      "ps.fonttype": 42
                  })

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 200

    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"

    for problem in config['datasets']:
        counter = 0

        pareto_fronts = {}

        for heuristic, renamed_heuristic in config['heuristics'].items():
            pareto_fronts[renamed_heuristic] = []
            if config["data_directory"] == "mlruns":
                fold_df = get_df(heuristic, problem)
            else:
                fold_df = get_csv_df(heuristic, problem)
            if fold_df is not None:
                counter += 1
                name = [renamed_heuristic] * fold_df.shape[0]
                current_res = fold_df.assign(Used_Representation=name)
                for path in current_res["artifact_uri"]:
                    path = path.split("suprb-experimentation/")[-1]
                    with open(os.path.join(path, "pareto_fronts.json")) as f:
                        pf = json.load(f)
                        pareto_fronts[renamed_heuristic].extend(pf[str(max(int(key) for key in pf.keys()))])

        for heuristic in pareto_fronts.keys():
            pareto_fronts[heuristic] = np.array(pareto_fronts[heuristic])

        fig_kde, axes_kde = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        fig_hex, axes_hex = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

        fig_kde.suptitle(config['datasets'][problem])
        fig_hex.suptitle(config['datasets'][problem])

        # ====== HEXBIN Plots ======

        for i, algo in enumerate(config["heuristics"].values()):
            algo_df = pd.DataFrame({
                "Normed Complexity": pareto_fronts[algo][:, 0],
                "Pseudo Accuracy": pareto_fronts[algo][:, 1]
            })

            hb = axes_hex[i].hexbin(
                algo_df['Normed Complexity'], algo_df['Pseudo Accuracy'],
                gridsize=30, cmap='viridis', extent=(0, 1, 0, 1)
            )
            axes_hex[i].set_title(f"{algo}")
            axes_hex[i].set_xlabel("Normed Complexity")

            fig_hex.colorbar(hb, ax=axes_hex[i], label='Density')

        fig_hex.supylabel("Pseudo Accuracy")
        plt.tight_layout()
        fig_hex.savefig(f"{final_output_dir}/{datasets_map[problem]}_hex.png")
        plt.close(fig_hex)

        # ====== KDE Plots ======
        for i, algo in enumerate(config["heuristics"].values()):
            algo_df = pd.DataFrame({
                "Normed Complexity": pareto_fronts[algo][:, 0],
                "Pseudo Accuracy": pareto_fronts[algo][:, 1]
            })


            sns.kdeplot(
                data=algo_df, x='Normed Complexity', y='Pseudo Accuracy',
                fill=True, cmap='Blues', ax=axes_kde[i],
                thresh=0.05, levels=100, clip=((0, 1), (0, 1))
            )
            axes_kde[i].set_title(f"{algo}")
            axes_kde[i].set_xlim(0, 1)
            axes_kde[i].set_ylim(0, 1)

        plt.tight_layout()
        fig_kde.savefig(f"{final_output_dir}/{datasets_map[problem]}_kde.png")
        plt.close(fig_kde)


if __name__ == '__main__':
    create_plots()
