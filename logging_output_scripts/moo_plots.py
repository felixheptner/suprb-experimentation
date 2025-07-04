from logging_output_scripts.utils import get_csv_df, get_normalized_df, check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import os
from utils import datasets_map
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"
hypervolume = "metrics.hypervolume"
spread = "metrics.spread"


def create_plots():
    """
    Creating Hexbin, Kernel density estimate, and Pareto Front cardinality histogram plots.
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
        first = True
        pareto_fronts = {}
        pareto_solutions = {}
        res_var = None # Initialize res_var here to ensure it's always defined

        for heuristic, renamed_heuristic in config['heuristics'].items():
            pareto_fronts[renamed_heuristic] = []
            pareto_solutions[renamed_heuristic] = []
            if config["data_directory"] == "mlruns":
                fold_df = get_df(heuristic, problem)
            else:
                fold_df = get_csv_df(heuristic, problem)
            if fold_df is not None:
                counter += 1
                name = [renamed_heuristic] * fold_df.shape[0]
                current_res = fold_df.assign(Used_Representation=name)
                if first:
                    first = False
                    res_var = current_res
                else:
                    # Adds additional column for plotting
                    res_var = pd.concat([res_var, current_res])
                for path in current_res["artifact_uri"]:
                    path = path.split("suprb-experimentation/")[-1]
                    with open(os.path.join(path, "pareto_fronts.json")) as f:
                        pf = json.load(f)
                        pf = pf[str(max(int(key) for key in pf.keys()))]
                        # The following is necessary as im a bit stupid and accidentally let the logger log ndarrays
                        # Into the dictionary which are not serializable by default and thus got turned into strings -.-
                        if isinstance(pf, str):
                            pf = re.sub(r"\s+", ",", pf)
                            pf = eval(pf)
                        pareto_fronts[renamed_heuristic].append(pf)
                        pareto_solutions[renamed_heuristic].extend(pf)

        for heuristic in pareto_fronts.keys():
            pareto_solutions[heuristic] = np.array(pareto_solutions[heuristic])

        # ================== Determine which heuristics have valid data for Iterations to Hypervolume ==================
        valid_ithv_heuristics = []
        # Ensure res_var is not None before proceeding
        if res_var is not None:
            for algo in config["heuristics"].values():
                algo_df = res_var.loc[res_var["Used_Representation"] == algo]
                iters = algo_df["metrics.sc_iterations"]
                # Condition for "ugly and redundant": all iteration values are the same
                if not iters.empty and iters.nunique() > 1: # Check if not empty and has more than 1 unique value
                    valid_ithv_heuristics.append(algo)
                else:
                    print(f"Skipping Iterations to Hypervolume plot for {algo} due to constant/empty iteration count.")

        # ================== Plotting Setup ==================
        # Always create these figures as they don't have conditional removal
        fig_hex, axes_hex = plt.subplots(1, len(config["heuristics"].values()), figsize=(18, 5), sharex=True,
                                         sharey=True, constrained_layout=True)
        fig_kde, axes_kde = plt.subplots(1, len(config["heuristics"].values()), figsize=(18, 5), sharex=True,
                                         sharey=True, constrained_layout=True)
        fig_hist, axes_hist = plt.subplots(1, len(config["heuristics"].values()), figsize=(18, 5), sharex=True,
                                         sharey=True, constrained_layout=True)

        fig_kde.suptitle(config['datasets'][problem])
        fig_hex.suptitle(config['datasets'][problem])
        fig_hist.suptitle(config['datasets'][problem])

        # ================== HEXBIN Plots ==================

        for i, algo in enumerate(config["heuristics"].values()):
            algo_df = pd.DataFrame({
                "Normed Complexity": pareto_solutions[algo][:, 0],
                "Pseudo Accuracy": pareto_solutions[algo][:, 1]
            })

            hb = axes_hex[i].hexbin(
                algo_df['Normed Complexity'], algo_df['Pseudo Accuracy'],
                gridsize=30, cmap='viridis', extent=(0, 1, 0, 1)
            )
            axes_hex[i].set_title(f"{algo}")
            axes_hex[i].set_xlabel("Normed Complexity")

            fig_hex.colorbar(hb, ax=axes_hex[i], label='Density')

        fig_hex.supylabel("Pseudo Accuracy")
        fig_hex.savefig(f"{final_output_dir}/{datasets_map[problem]}_hex.png")
        plt.close(fig_hex)

        # ================== KDE Plots ==================

        for i, algo in enumerate(config["heuristics"].values()):
            algo_df = pd.DataFrame({
                "Normed Complexity": pareto_solutions[algo][:, 0],
                "Pseudo Accuracy": pareto_solutions[algo][:, 1]
            })

            sns.kdeplot(
                data=algo_df, x='Normed Complexity', y='Pseudo Accuracy',
                fill=True, cmap='Blues', ax=axes_kde[i],
                thresh=0.05, levels=100, clip=((0, 1), (0, 1))
            )
            axes_kde[i].set_title(f"{algo}")
            axes_kde[i].set_xlim(0, 1)
            axes_kde[i].set_ylim(0, 1)

        fig_kde.savefig(f"{final_output_dir}/{datasets_map[problem]}_kde.png")
        plt.close(fig_kde)

        # ================== |Pareto Front| Histograms ==================

        for i, algo in enumerate(config["heuristics"].values()):
            pf_lengths = [len(pf) for pf in pareto_fronts[algo]]
            axes_hist[i].hist(pf_lengths, bins=np.arange(1, 33), align='left', rwidth=0.9)
            axes_hist[i].set_title(f"{algo}")
            axes_hist[i].set_xlabel(f"Pareto Front Length")

        fig_hist.supylabel("Cardinalities") # This was `fig_hex.supylabel` before, changed to `fig_hist`
        fig_hist.savefig((f"{final_output_dir}/{datasets_map[problem]}_hist.png"))
        plt.close(fig_hist)

        # ================== Iterations to Hypervolume (Conditional Plotting) ==================
        if valid_ithv_heuristics: # Only create the figure if there are valid heuristics to plot
            fig_ithv, axes_ithv = plt.subplots(1, len(valid_ithv_heuristics), figsize=(6 * len(valid_ithv_heuristics), 5), # Adjust figsize dynamically
                                               sharex=True, sharey=True, constrained_layout=True)
            # Ensure axes_ithv is always an array, even if there's only one subplot
            if len(valid_ithv_heuristics) == 1:
                axes_ithv = [axes_ithv]

            fig_ithv.suptitle(config['datasets'][problem])

            for i, algo in enumerate(valid_ithv_heuristics):
                algo_df = res_var.loc[res_var["Used_Representation"] == algo]
                iters = algo_df["metrics.sc_iterations"]
                hv = algo_df["metrics.hypervolume"]
                algo_df = pd.DataFrame({
                    "Iterations": iters,
                    "Hypervolume": hv
                })
                sns.scatterplot(data=algo_df, x="Iterations", y="Hypervolume", ax=axes_ithv[i])

                # Fit regression line
                X = algo_df["Iterations"].values.reshape(-1, 1)
                y = algo_df["Hypervolume"].values
                # Check for NaNs and infs before fitting, and ensure X has enough unique points
                if not np.isnan(np.sum(X)) and not np.isinf(np.sum(X)) and len(np.unique(X)) > 1:
                    reg = LinearRegression().fit(X, y)
                    axes_ithv[i].plot(X, reg.predict(X), color='red', linewidth=2)
                else:
                    print(f"Skipping regression line for {algo} due to insufficient data or constant X.")


                axes_ithv[i].set_title(f"{algo}")
            fig_ithv.savefig(f"{final_output_dir}/{datasets_map[problem]}_ithv.png")
            plt.close(fig_ithv) # Close fig_ithv, not fig_kde

        else:
            print(f"No valid data to plot Iterations to Hypervolume for problem: {problem}. Skipping this plot.")


if __name__ == '__main__':
    create_plots()