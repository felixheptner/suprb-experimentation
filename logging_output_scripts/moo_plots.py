from logging_output_scripts.utils import get_csv_df, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import os
from utils import datasets_map
from sklearn.linear_model import LinearRegression
from suprb.logging.metrics import spread as metric_spread
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"
hypervolume = "metrics.hypervolume"
spread = "metrics.spread"

ga_baselines = {"Baseline c:ga32": ("x", "red"), "Baseline c:ga64": ("+", "green"), "Baseline c:ga_no_tuning": ("1", "orange")}

def confidence_ellipse(mean, cov, ax, n_std=1.96, color='red', **kwargs):
    """
    https://de.matplotlib.net/stable/gallery/statistics/confidence_ellipse.html#the-plotting-function-itself
    http://www.econ.uiuc.edu/~roger/courses/471/lectures/L5.pdf

    Create a plot of the covariance confidence ellipse with mean and cov*.

    Parameters
    ----------
    mean: array-like, shape (2,)
        mean of the data point
    cov: array-like, shape (2,2)
        covariance matrix
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=color, edgecolor=color, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    ellipse.set_edgecolor(color)
    ellipse.set_facecolor((ellipse.get_facecolor()[0], ellipse.get_facecolor()[1], ellipse.get_facecolor()[2], 0.2)) # 20% alpha for fill
    ax.add_patch(ellipse)
    return ellipse

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

        # Filter out soo comparison and compute data for soo overlays
        moo_heuristics = [config["heuristics"][algo] for algo in config["heuristics"].keys() if not algo in ga_baselines.keys()]
        soo_heuristics = [(config["heuristics"][algo], ga_baselines[algo]) for algo in config["heuristics"].keys() if algo in ga_baselines.keys()]
        soo_averages = {}
        soo_standard_devs = {}
        for algo, style in soo_heuristics:
            if len(pareto_solutions[algo]) > 0:
                soo_averages[algo] = (np.mean(pareto_solutions[algo], axis=0), style)
                soo_standard_devs[algo] = (np.cov(pareto_solutions[algo].T), style)

        # Determine which heuristics have valid data for Iterations to Hypervolume
        valid_ithv_heuristics = []
        # Ensure res_var is not None before proceeding
        if res_var is not None:
            for algo in moo_heuristics:
                algo_df = res_var.loc[res_var["Used_Representation"] == algo]
                iters = algo_df["metrics.sc_iterations"]
                # Condition for "ugly and redundant": all iteration values are the same
                if not iters.empty and iters.nunique() > 1: # Check if not empty and has more than 1 unique value
                    valid_ithv_heuristics.append(algo)
                else:
                    print(f"Skipping Iterations to Hypervolume plot for {algo} due to constant/empty iteration count.")

        # Plotting Setup
        n_algs = len(moo_heuristics)
        if n_algs <= 3:
            n_cols = n_algs
            n_rows = 1
        elif n_algs == 4:
            n_cols = n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_algs - 1) // 3 + 1

        # Always create these figures as they don't have conditional removal
        fig_hex, axes_hex = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True,
                                         sharey=True, constrained_layout=True)
        fig_kde, axes_kde = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True,
                                         sharey=True, constrained_layout=True)
        fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True,
                                         sharey=True, constrained_layout=True)

        # Always make axes a two dimensional ndarray
        if n_rows == 1:
            axes_hex = axes_hex[None, :]
            axes_kde = axes_kde[None, :]
            axes_hist = axes_hist[None, :]

        fig_kde.suptitle(config['datasets'][problem])
        fig_hex.suptitle(config['datasets'][problem])
        fig_hist.suptitle(config['datasets'][problem])

        # ================== HEXBIN Plots ==================
        soo_handles = []
        for i, algo in enumerate(moo_heuristics):
            algo_df = pd.DataFrame({
                "Normed Complexity": pareto_solutions[algo][:, 0],
                "Pseudo Accuracy": pareto_solutions[algo][:, 1]
            })

            hb = axes_hex[i // n_cols, i % n_cols].hexbin(
                algo_df['Normed Complexity'], algo_df['Pseudo Accuracy'],
                gridsize=30, cmap='Blues', extent=(0, 1, 0, 1), mincnt=1,
            )

            axes_hex[i // n_cols, i % n_cols].set_title(f"{algo}")

            # Plot SOO comparison points
            for (soo_algo, value) in soo_averages.items():
                avg, style = value
                cov = soo_standard_devs[soo_algo][0]
                axes_hex[i // n_cols, i % n_cols].plot(
                    avg[0], avg[1], marker=style[0], markersize=6,
                    markeredgewidth=1.2, color=style[1],
                    linestyle='None', label=f"{soo_algo} Mean"
                )
                confidence_ellipse(avg, cov, axes_hex[i // n_cols, i % n_cols], n_std=1.96,
                                   color=style[1], label=f"{soo_algo} 95% CI", alpha=0.2)

            axes_hex[i // n_cols, i % n_cols].legend(fontsize=14, loc="upper right")
            axes_hex[i // n_cols, i % n_cols].set_xlabel("$f_1$")
            fig_hex.colorbar(hb, ax=axes_hex[i // n_cols, i % n_cols], label='Density')

        fig_hex.supylabel("$f_2$")
        fig_hex.savefig(f"{final_output_dir}/{datasets_map[problem]}_hex.png")
        plt.close(fig_hex)

        # ================== KDE Plots ==================

        soo_handles
        for i, algo in enumerate(moo_heuristics):
            algo_df = pd.DataFrame({
                "Normed Complexity": pareto_solutions[algo][:, 0],
                "Pseudo Accuracy": pareto_solutions[algo][:, 1]
            })

            sns.kdeplot(
                data=algo_df, x='Normed Complexity', y='Pseudo Accuracy',
                fill=True, cmap='Blues', ax=axes_kde[i // n_cols, i % n_cols],
                thresh=0.05, levels=100, clip=((0, 1), (0, 1))
            )
            axes_kde[i // n_cols, i % n_cols].set_title(f"{algo}")
            axes_kde[i // n_cols, i % n_cols].set_xlim(0, 1)
            axes_kde[i // n_cols, i % n_cols].set_ylim(0, 1)

        fig_kde.savefig(f"{final_output_dir}/{datasets_map[problem]}_kde.png")
        plt.close(fig_kde)

        # ================== |Pareto Front| Histograms ==================

        for i, algo in enumerate(moo_heuristics):
            pf_lengths = [len(pf) for pf in pareto_fronts[algo]]
            max_length = max(pf_lengths)
            axes_hist[i // n_cols, i % n_cols].hist(pf_lengths, bins=np.arange(1, max(33, max_length + 1)), align='left', rwidth=0.9)
            axes_hist[i // n_cols, i % n_cols].set_title(f"{algo}")
            axes_hist[i // n_cols, i % n_cols].set_xlabel(f"Pareto Front Length")

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

        # ================== Spread Swarm Plot ==================

        spread_df = []

        for algo in pareto_fronts.keys():
            for front in pareto_fronts[algo]:
                front_np = np.array(front)
                sp = metric_spread(front_np)
                spread_df.append({
                    "Used_Representation": algo,
                    "Spread": sp
                })

        spread_df = pd.DataFrame(spread_df)

        # Swarm plot
        fig_swarm, ax_swarm = plt.subplots(dpi=400)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.22)

        sns.swarmplot(data=spread_df, x="Used_Representation", y="Spread", ax=ax_swarm, size=3)
        ax_swarm.set_title(f"{config['datasets'][problem]}", style="italic", fontsize=14)
        ax_swarm.set_ylabel("Spread", fontsize=18, weight="bold")
        ax_swarm.set_xlabel("")
        ax_swarm.tick_params(axis='x', rotation=15)
        y_min = max(0, min(ax_swarm.get_yticks()))
        y_max = max(ax_swarm.get_yticks())
        num_ticks = 7
        y_tick_positions = np.linspace(y_min, y_max, num_ticks)
        y_tick_positions = np.round(y_tick_positions, 3)

        ax_swarm.set_ylim(y_min, y_max)
        ax_swarm.set_yticks(y_tick_positions)
        ax_swarm.set_yticklabels([f'{x:.3g}' for x in y_tick_positions])
        plt.xticks(rotation=15, ha='right', fontsize=12)
        plt.tight_layout()
        fig_swarm.savefig(f"{final_output_dir}/{datasets_map[problem]}_swarm_spread.png")
        plt.close(fig_swarm)

        # ================== Spread Violin Plot ==================
        fig_violin, ax_violin = plt.subplots(dpi=400)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.22)

        sns.violinplot(data=spread_df, x="Used_Representation", y="Spread", ax=ax_violin, inner="box", size=3)
        ax_violin.set_title(f"{config['datasets'][problem]}", style="italic", fontsize=14)
        ax_violin.set_ylabel("Spread", fontsize=18, weight="bold")
        ax_violin.set_xlabel("")
        ax_violin.tick_params(axis='x', rotation=15)
        ax_swarm.set_ylim(y_min, y_max)
        ax_swarm.set_yticks(y_tick_positions)
        ax_swarm.set_yticklabels([f'{x:.3g}' for x in y_tick_positions])
        plt.xticks(rotation=15, ha='right', fontsize=12)
        plt.tight_layout()
        fig_violin.savefig(f"{final_output_dir}/{datasets_map[problem]}_violin_spread.png")
        plt.close(fig_violin)

if __name__ == '__main__':
    create_plots()