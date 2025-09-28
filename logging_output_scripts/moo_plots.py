from logging_output_scripts.utils import get_csv_df, get_df, get_csv_root_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import os
from utils import datasets_map
from sklearn.linear_model import LinearRegression
from suprb.logging.metrics import spread as metric_spread, hypervolume as metric_hypervolume
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from typing import Dict, List, Tuple, Any, Union, Optional
import ast

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"
hypervolume = "metrics.hypervolume"
spread = "metrics.spread"

ga_baselines = {"Baseline c:ga32": ("x", "red"),
                "Baseline c:ga64": ("+", "green"),
                }

# Mapping raw parameter keys to nicer LaTeX display names
PARAM_NAME_MAP = {
    "rule_discovery__mutation__sigma": "\\acs{RD} $\\sigma_{mut}$",
    "rule_discovery__init__fitness__alpha": "\\acs{RD} $\\alpha_{init}$",
    "solution_composition__crossover": "\\acs{SC} Crossover",
    "solution_composition__crossover__n": "\\acs{SC} $n_{cross}$",
    "solution_composition__mutation__mutation_rate": "\\acs{SC} $\\sigma_{mut}$",
    "solution_composition__sampler__a": "$a$",
    "solution_composition__sampler__b": "$b$",
    "solution_composition__algorithm_1__crossover": "\\acs{GA} Crossover",
    "solution_composition__algorithm_1__crossover__n": "\\acs{GA} $n_{cross}$",
    "solution_composition__algorithm_1__mutation_rate": "\\acs{GA} $\\sigma_{mut}$",
    "solution_composition__algorithm_1__selection__k": "\\acs{GA} $k_{sel}$",
    "solution_composition__algorithm_2__crossover": "\\acs{MOO} Crossover",
    "solution_composition__algorithm_2__crossover__n": "\\acs{MOO} $n_{cross}$",
    "solution_composition__algorithm_2__mutation_rate": "\\acs{MOO} $\\sigma_{mut}$",
    "solution_composition__algorithm_2__selection__k": "\\acs{MOO} $k_{sel}$",
}


def _esc_tex(s: str) -> str:
    return str(s).replace('_', r'\_')

def f_1_pareto_sacrifice(moo_front: np.ndarray, ga_front: np.ndarray) -> float:
    """
    Compute the f_1 distance from the one GA solution to the solution along the MOO Pareto front with the minimum
    f_1 value whose f2 value is smaller than the GA solution's f_2 value. If no MOO solution has f_2 smaller than the GA solution's f_2,
    choose the MOO solution with the smallest f_2.
    """
    if len(moo_front) == 0 or len(ga_front) == 0:
        return np.nan
    ga_sol = ga_front[0]
    ga_f2 = ga_sol[1]
    # Filter MOO front to only those with f2 less than ga_f2
    filtered_moo = moo_front[moo_front[:, 1] <= ga_f2]
    if len(filtered_moo) == 0:
        closest_moo = moo_front[np.argmin(moo_front[:, 1])]
        return np.nan
    else:
        closest_moo = filtered_moo[np.argmin(filtered_moo[:, 0])]
        return closest_moo[0] - ga_sol[0]


def generate_pareto_sacrifice_undefined_table(
    per_problem_nan: Dict[str, Dict[str, Optional[float]]],
    cfg: Dict[str, Any],
    moo_algos: List[str],
    final_output_dir: str
) -> None:
    """
    One LaTeX table: rows=datasets, cols=moo_algos, values=\% of NaNs in $f_1$ Pareto sacrifice
    """
    os.makedirs(final_output_dir, exist_ok=True)
    header = ["Dataset"] + moo_algos
    col_format = "l" + "c" * len(moo_algos)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Percentage of \textit{Pareto Sacrifice undefined} (NaN) per algorithm and dataset.}",
        r"\label{tab:pareto_sacrifice_undefined_" + "".join(moo_algos).lower() + "}",
        r"\begin{tabular}{" + col_format + r"}",
        r"\hline",
        " & ".join([_esc_tex(h) for h in header]) + r" \\",
        r"\hline"
    ]

    # Keep dataset order from cfg
    for problem in cfg["datasets"]:
        ds_name = _esc_tex(datasets_map[problem].upper())
        stats = per_problem_nan.get(problem, {})
        row_vals: List[str] = []
        for algo in moo_algos:
            v = stats.get(algo, None)
            cell = "N/A" if v is None or np.isnan(v) else f"{v:.1f}~\\%"
            row_vals.append(cell)
        lines.append(ds_name + " & " + " & ".join(row_vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    out_path = os.path.join(final_output_dir, "pareto_sacrifice_undefined.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def generate_tuning_tables(tuning_params: Dict[str, Dict[str, Dict]],
                           config: Dict[str, Any],
                           final_output_dir: str) -> None:
    """
    Generate one LaTeX table per heuristic (renamed) showing tuned parameter values across datasets.
    Rows: datasets
    Columns: union of all parameter names observed for that heuristic across datasets.
    Missing values are filled with N/A.
    """
    os.makedirs(final_output_dir, exist_ok=True)
    heuristic_display_map = config["heuristics"]

    # Collect all heuristic keys encountered
    heuristic_keys = set()
    for problem_dict in tuning_params.values():
        heuristic_keys.update(problem_dict.keys())

    for heuristic_key in sorted(heuristic_keys):
        display_name = heuristic_display_map.get(heuristic_key, heuristic_key)

        # Build an ordered union of parameter keys (order of first appearance)
        seen = {}
        for problem in config["datasets"]:
            params_dict = tuning_params.get(problem, {}).get(heuristic_key, {})
            for k in params_dict.keys():
                if k not in seen:
                    seen[k] = None
        if not seen:
            continue
        param_keys = list(seen.keys())

        param_keys_display = [PARAM_NAME_MAP.get(k, k) for k in param_keys]
        param_keys, param_keys_display = zip(*sorted(zip(param_keys, param_keys_display), key=lambda x: x[1]))
        param_keys = list(param_keys)
        param_keys_display = list(param_keys_display)
        header_cols = ["Dataset"] + param_keys_display
        col_format = "l" + "c" * len(param_keys)

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Tuned parameters for " + _esc_tex(display_name) + r"}",
            r"\adjustbox{max width=\textwidth}{"
            r"\begin{tabular}{" + col_format + r"}",
            r"\hline",
            " & ".join(header_cols) + r" \\",
            r"\hline"
        ]

        for problem in config["datasets"]:
            dataset_name = datasets_map[problem].upper()
            params_dict = tuning_params.get(problem, {}).get(heuristic_key, {})
            row_vals = []
            for k in param_keys:
                if k in params_dict:
                    v = params_dict[k]
                    if isinstance(v, float):
                        v = f"{v:.4f}"
                else:
                    v = "N/A"
                row_vals.append(_esc_tex(v))
            lines.append(_esc_tex(dataset_name) + " & " + " & ".join(row_vals) + r" \\")
        lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]

        content = "\n".join(lines)
        file_safe = display_name.lower().replace(" ", "_")
        out_path = os.path.join(final_output_dir, f"{file_safe}_tuned_params.tex")
        with open(out_path, "w") as f:
            f.write(content)

def configure_style() -> None:
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


def load_config(path: str = 'logging_output_scripts/config.json') -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


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
    ellipse.set_facecolor((ellipse.get_facecolor()[0], ellipse.get_facecolor()[1], ellipse.get_facecolor()[2], 0.2))  # 20% alpha for fill
    ax.add_patch(ellipse)
    return ellipse


def load_fold_dataframe(heuristic: str, problem: str, cfg: Dict[str, Any]) -> Union[pd.DataFrame, None]:
    if cfg["data_directory"] == "mlruns":
        return get_df(heuristic, problem)
    return get_csv_df(heuristic, problem), get_csv_root_df(heuristic, problem)

def load_roof_dataframe(heuristic: str, problem: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    if cfg["data_directory"] == "mlruns":
        return get_df(heuristic, problem)
    return get_csv_df(heuristic, problem)


def process_artifact_paths(current_res: pd.DataFrame,
                           renamed_heuristic: str,
                           train_pareto_fronts: Dict[str, List[List]],
                           train_pareto_solutions: Dict[str, List[List[float]]],
                           test_pareto_fronts: Dict[str, List[List]],
                           test_pareto_solutions: Dict[str, List[List[float]]]) -> None:
    for path in current_res["artifact_uri"]:
        path = path.split("suprb-experimentation/")[-1]
        with open(os.path.join(path, "pareto_fronts.json")) as f:
            train_pf = json.load(f)
            train_pf = train_pf[str(max(int(key) for key in train_pf.keys()))]
            train_pareto_fronts[renamed_heuristic].append(train_pf)
            train_pareto_solutions[renamed_heuristic].extend(train_pf)
        with open(os.path.join(path, "test_pareto_front.json")) as f:
            test_pf = json.load(f)
            test_pf = test_pf["test_pf_fitness"]
            test_pareto_fronts[renamed_heuristic].append(test_pf)
            test_pareto_solutions[renamed_heuristic].extend(test_pf)


def prepare_soo_stats(pareto_solutions: Dict[str, np.ndarray],
                      cfg: Dict[str, Any]) -> Tuple[Dict[str, Tuple[np.ndarray, Tuple[str, str]]],
                                                    Dict[str, Tuple[np.ndarray, Tuple[str, str]]]]:
    soo_heuristics = [(cfg["heuristics"][algo], ga_baselines[algo]) for algo in cfg["heuristics"].keys() if algo in ga_baselines.keys()]
    soo_averages: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    soo_standard_devs: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    for algo, style in soo_heuristics:
        if len(pareto_solutions[algo]) > 0:
            soo_averages[algo] = (np.mean(pareto_solutions[algo], axis=0), style)
            soo_standard_devs[algo] = (np.cov(pareto_solutions[algo].T), style)
    return soo_averages, soo_standard_devs


def determine_layout(n_algs: int) -> Tuple[int, int]:
    # Plotting Setup
    if n_algs <= 3:
        return n_algs, 1
    if n_algs == 4:
        return 2, 2
    n_cols = 3
    n_rows = (n_algs - 1) // 3 + 1
    return n_cols, n_rows


def plot_hexbin(moo_heuristics: List[str],
                pareto_solutions: Dict[str, np.ndarray],
                soo_averages: Dict[str, Tuple[np.ndarray, Tuple[str, str]]],
                soo_standard_devs: Dict[str, Tuple[np.ndarray, Tuple[str, str]]],
                n_cols: int, n_rows: int,
                title: str,
                final_output_dir: str,
                dataset_key: str,
                plot_type: str = "test") -> None:
    """
    Create hexbin plot for Pareto solutions.

    Parameters:
    -----------
    pareto_solutions: Dict[str, np.ndarray]
        Dictionary mapping algorithm names to their Pareto solution arrays
    plot_type: str
        String identifier to add to plot title and filename (e.g., "test", "train")
    """
    fig_hex, axes_hex = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True,
                                     sharey=True, constrained_layout=True, squeeze=False)

    # Add plot_type to title if it's not "test" (for backward compatibility)
    display_title = f"{title}" if plot_type == "test" else f"{title} ({plot_type.capitalize()})"
    fig_hex.suptitle(display_title)

    for i, algo in enumerate(moo_heuristics):
        algo_df = pd.DataFrame({
            "Normed Complexity": pareto_solutions[algo][:, 0],
            "Pseudo Accuracy": pareto_solutions[algo][:, 1]
        })
        hb = axes_hex[i % n_rows, i // n_rows].hexbin(
            algo_df['Normed Complexity'], algo_df['Pseudo Accuracy'],
            gridsize=30, cmap='Blues' if plot_type == "test" else 'Oranges',
            extent=(0, 1, 0, 1), mincnt=1,
        )
        axes_hex[i % n_rows, i // n_rows].set_title(f"{algo}")
        # Plot SOO comparison points
        for (soo_algo, value) in soo_averages.items():
            avg, style = value
            cov = soo_standard_devs[soo_algo][0]
            axes_hex[i % n_rows, i // n_rows].plot(
                avg[0], avg[1], marker=style[0], markersize=6,
                markeredgewidth=1.2, color=style[1],
                linestyle='None', label=f"{soo_algo} Mean"
            )
            confidence_ellipse(avg, cov, axes_hex[i % n_rows, i // n_rows], n_std=1.96,
                               color=style[1], label=f"{soo_algo} 95% CI", alpha=0.2)
        axes_hex[i % n_rows, i // n_rows].legend(fontsize=14, loc="upper right")
        axes_hex[i % n_rows, i // n_rows].set_xlabel("$f_1$")
        fig_hex.colorbar(hb, ax=axes_hex[i % n_rows, i // n_rows], label='Density')
    fig_hex.supylabel("$f_2$")

    # Create filename based on plot_type
    filename = f"{dataset_key}_hex{'' if plot_type == 'test' else '_' + plot_type}.png"
    fig_hex.savefig(f"{final_output_dir}/{filename}")
    plt.close(fig_hex)


def plot_mean_shift_arrows(moo_heuristics: List[str],
                           train_pareto_solutions: Dict[str, np.ndarray],
                           test_pareto_solutions: Dict[str, np.ndarray],
                           n_cols: int, n_rows: int,
                           title: str, final_output_dir: str, dataset_key: str,
                           n_bins: int = 20) -> None:
    """
    Plot mean shift arrows of f1 between training and test per f0 bin.
    Scatter + arrows are centered at the bin midpoint.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    fig.suptitle(f"{title} (Mean shift arrows)")

    for i, algo in enumerate(moo_heuristics):
        ax = axes[i // n_cols, i % n_cols]
        train = train_pareto_solutions[algo]
        test = test_pareto_solutions[algo]

        if train.shape[0] == 0 or test.shape[0] == 0:
            ax.set_title(f"{algo} (no data)")
            continue

        f0 = train[:, 0]
        f1_train = train[:, 1]
        f1_test = test[:, 1]

        bins = np.linspace(f0.min(), f0.max(), n_bins + 1)
        bin_indices = np.digitize(f0, bins) - 1

        arrow_data = []
        for j in range(n_bins):
            mask = bin_indices == j
            if np.any(mask):
                bin_center = (bins[j] + bins[j + 1]) / 2
                mean_train = f1_train[mask].mean()
                mean_test = f1_test[mask].mean()
                arrow_data.append((bin_center, mean_train, mean_test))

        for bin_center, mean_train, mean_test in arrow_data:
            ax.arrow(bin_center, mean_train,
                     0, mean_test - mean_train,
                     head_width=0.01, head_length=0.01,
                     length_includes_head=True,
                     color="steelblue", alpha=0.8)
        ax.scatter([d[0] for d in arrow_data], [d[1] for d in arrow_data],
                   label="fTrain mean $f_1$", color="blue", marker="o", s=30)
        ax.scatter([d[0] for d in arrow_data], [d[2] for d in arrow_data],
                   label="Test mean $f_1$", color="red", marker="x", s=40)

        ax.set_title(f"{algo}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)

    fig.supxlabel("$f_0$")
    fig.supylabel("$f_1$")
    filename = f"{dataset_key}_mean_shift.png"
    fig.savefig(f"{final_output_dir}/{filename}")
    plt.close(fig)


def plot_hist(moo_heuristics: List[str],
              train_pareto_fronts: Dict[str, List[List]],
              n_cols: int, n_rows: int,
              title: str, final_output_dir: str, dataset_key: str) -> None:
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True,
                                       sharey=True, constrained_layout=True, squeeze=False)
    fig_hist.suptitle(title)
    for i, algo in enumerate(moo_heuristics):
        pf_lengths = [len(pf) for pf in train_pareto_fronts[algo]]
        if not pf_lengths:
            continue
        max_length = max(pf_lengths)
        axes_hist[i // n_cols, i % n_cols].hist(pf_lengths, bins=np.arange(1, max(33, max_length + 1)), align='left', rwidth=0.9)
        axes_hist[i // n_cols, i % n_cols].set_title(f"{algo}")
        axes_hist[i // n_cols, i % n_cols].set_xlabel(f"Cardinalities")
    fig_hist.supylabel("# of Pareto fronts")
    fig_hist.savefig((f"{final_output_dir}/{dataset_key}_hist.png"))
    plt.close(fig_hist)


def plot_iterations_hv(res_var: pd.DataFrame,
                       moo_heuristics: List[str],
                       final_output_dir: str,
                       dataset_key: str,
                       title: str) -> None:
    # Determine which heuristics have valid data for Iterations to Hypervolume
    valid_ithv_heuristics: List[str] = []
    # Ensure res_var is not None before proceeding
    if res_var is not None:
        for algo in moo_heuristics:
            algo_df = res_var.loc[res_var["Used_Representation"] == algo]
            iters = algo_df["metrics.sc_iterations"]
            # Condition for "ugly and redundant": all iteration values are the same
            if not iters.empty and iters.nunique() > 1:  # Check if not empty and has more than 1 unique value
                valid_ithv_heuristics.append(algo)
            else:
                print(f"Skipping Iterations to Hypervolume plot for {algo} due to constant/empty iteration count.")
    if not valid_ithv_heuristics:
        print(f"No valid data to plot Iterations to Hypervolume for problem: {dataset_key}. Skipping this plot.")
        return
    fig_ithv, axes_ithv = plt.subplots(1, len(valid_ithv_heuristics),
                                       figsize=(6 * len(valid_ithv_heuristics), 5),  # Adjust figsize dynamically
                                       sharex=True, sharey=True, constrained_layout=True)
    # Ensure axes_ithv is always an array, even if there's only one subplot
    if len(valid_ithv_heuristics) == 1:
        axes_ithv = [axes_ithv]
    fig_ithv.suptitle(title)
    for i, algo in enumerate(valid_ithv_heuristics):
        algo_df = res_var.loc[res_var["Used_Representation"] == algo]
        iters = algo_df["metrics.sc_iterations"]
        hv = algo_df["metrics.test_hypervolume"]
        algo_df_plot = pd.DataFrame({
            "Iterations": iters,
            "Hypervolume": hv
        })
        sns.scatterplot(data=algo_df_plot, x="Iterations", y="Hypervolume", ax=axes_ithv[i])
        # Fit regression line
        X = algo_df_plot["Iterations"].values.reshape(-1, 1)
        y = algo_df_plot["Hypervolume"].values
        # Check for NaNs and infs before fitting, and ensure X has enough unique points
        if not np.isnan(np.sum(X)) and not np.isinf(np.sum(X)) and len(np.unique(X)) > 1:
            reg = LinearRegression().fit(X, y)
            axes_ithv[i].plot(X, reg.predict(X), color='red', linewidth=2)
        else:
            print(f"Skipping regression line for {algo} due to insufficient data or constant X.")
        axes_ithv[i].set_title(f"{algo}")
    fig_ithv.savefig(f"{final_output_dir}/{dataset_key}_ithv.png")
    plt.close(fig_ithv)  # Close fig_ithv, not fig_kde


def compute_metric_dataframe(pareto_fronts: Dict[str, List[List]], metric: callable, name: str, *args, **kwargs) -> pd.DataFrame:
    m_rows = []
    for algo in pareto_fronts.keys():
        for idx, front in enumerate(pareto_fronts[algo]):
            front_np = np.array(front)
            m = metric(front_np, *args, **kwargs)
            m_rows.append({
                "Used_Representation": algo,
                name: m
            })
    return pd.DataFrame(m_rows)

def compute_ga_comparison_dataframe(pareto_fronts: Dict[str, List[List]], ref_name:str):
    m_rows = []
    ref_fronts = pareto_fronts[ref_name]
    for algo in pareto_fronts.keys():
        for idx, front in enumerate(pareto_fronts[algo]):
            front_np = np.array(front)
            ref_front_np = np.array(ref_fronts[idx])
            m = f_1_pareto_sacrifice(front_np, ref_front_np)
            m_rows.append({
                "Used_Representation": algo,
                f"$f_1$ Pareto sacrifice to {ref_name}": m
            })
    return pd.DataFrame(m_rows)

def plot_violin_metric(metric_df: pd.DataFrame,
                             cfg: Dict[str, Any],
                             problem: str,
                             final_output_dir: str,
                             name: str,
                             allowed_algos: List[str] | None = None) -> None:
    # Filter to only MOO heuristics if provided
    if allowed_algos is not None:
        metric_df = metric_df[metric_df["Used_Representation"].isin(allowed_algos)]

    # Filter out Nan values from the metric column (turns out to be unnecessary)
    metric_df = metric_df[metric_df[name].notna()]
    # ================== Spread Violin Plot ==================
    fig_violin, ax_violin = plt.subplots(dpi=400)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.22)
    sns.violinplot(data=metric_df, x="Used_Representation", y=name, ax=ax_violin, inner="box", palette="tab10",
                   hue="Used_Representation", legend=False)
    ax_violin.set_title(f"{cfg['datasets'][problem]}", style="italic", fontsize=14)
    ax_violin.set_ylabel(name, fontsize=18, weight="bold")
    ax_violin.set_xlabel("")
    ax_violin.tick_params(axis='x', rotation=15)#
    y_min = min(ax_violin.get_yticks())
    y_max = max(ax_violin.get_yticks())
    num_ticks = 7
    y_tick_positions = np.linspace(y_min, y_max, num_ticks)
    if y_min <= 0 <= y_max:

        y_tick_positions = np.concat([y_tick_positions, [0]])
    y_tick_positions = np.round(y_tick_positions, 3)
    ax_violin.set_ylim(y_min, y_max)
    ax_violin.set_yticks(y_tick_positions)
    ax_violin.set_yticklabels([f'{x:.3g}' for x in y_tick_positions])
    plt.xticks(rotation=15, ha='right', fontsize=12)
    plt.tight_layout()
    fig_violin.savefig(f"{final_output_dir}/{datasets_map[problem]}_violin_{name}.png")
    plt.close(fig_violin)


def prepare_reference_stats(pareto_solutions: Dict[str, np.ndarray],
                            cfg: Dict[str, Any]) -> Tuple[
                                Dict[str, Tuple[np.ndarray, Tuple[str, str]]],
                                Dict[str, Tuple[np.ndarray, Tuple[str, str]]]]:
    """
    Prepares mean points and covariance (for confidence ellipses) of reference (SOO) heuristics.
    Extracts soo_pareto_fronts
    Uses cfg['reference_heuristics'] (raw_name -> display_name).
    """
    ref_cfg = cfg.get("reference_heuristics", {})
    averages: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    covs: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    for raw_name, display_name in ref_cfg.items():
        style = ga_baselines.get(raw_name, ("o", "black"))
        if display_name in pareto_solutions and len(pareto_solutions[display_name]) > 0:
            data = pareto_solutions[display_name]
            averages[display_name] = (np.mean(data, axis=0), style)
            # Need at least 2 points for covariance
            if data.shape[0] > 1:
                covs[display_name] = (np.cov(data.T), style)
            else:
                # Fallback: zero covariance
                covs[display_name] = (np.zeros((2, 2)), style)
    return averages, covs


def compute_nan_percentage_per_algo(metric_df: pd.DataFrame,
                                    value_col: str,
                                    algo_list: List[str]) -> Dict[str, Optional[float]]:
    """
    Returns a dict algo -> percentage of NaNs in value_col (0..100) or None if no rows for algo.
    """
    out: Dict[str, Optional[float]] = {}
    if metric_df is None or metric_df.empty or value_col not in metric_df.columns:
        for a in algo_list:
            out[a] = None
        return out
    for algo in algo_list:
        sub = metric_df[metric_df["Used_Representation"] == algo]
        total = int(sub.shape[0])
        if total == 0:
            out[algo] = None
            continue
        n_nans = int(sub[value_col].isna().sum())
        out[algo] = 100.0 * n_nans / float(total)
    return out


def create_plots():
    configure_style()
    config = load_config()
    final_output_dir = f"{config['output_directory']}"

    tuning_info: Dict[str, Dict[str, Dict]] = {}

    # Merge MOO and reference heuristics for data loading
    ref_heurs = config.get('reference_heuristics', {})
    all_heuristics: Dict[str, str] = {**config['heuristics'], **ref_heurs}

    # Accumulator for Pareto Sacrifice undefined table
    pareto_sacrifice_nan_stats: Dict[str, Dict[str, Optional[float]]] = {}

    for problem in config['datasets']:
        counter = 0
        first = True
        train_pareto_fronts: Dict[str, List[List]] = {}
        train_pareto_solutions: Dict[str, List[List[float]]] = {}
        test_pareto_fronts: Dict[str, List[List]] = {}
        test_pareto_solutions: Dict[str, List[List[float]]] = {}
        res_var = None
        tuning_info[problem] = {}

        # Load runs for all heuristics (MOO and reference)
        for heuristic, renamed_heuristic in all_heuristics.items():
            train_pareto_fronts[renamed_heuristic] = []
            train_pareto_solutions[renamed_heuristic] = []
            test_pareto_fronts[renamed_heuristic] = []
            test_pareto_solutions[renamed_heuristic] = []

            fold_df, root_df = load_fold_dataframe(heuristic, problem, config)
            if root_df is not None and not root_df.empty:
                string_dict = root_df["params.tuned_params"].iloc[0]
                string_dict = re.sub(r'([A-Za-z_]\w*)\([^)]*\)', r"'\1'", string_dict)
                try:
                    tuning_info[problem][heuristic] = ast.literal_eval(string_dict)
                except Exception:
                    tuning_info[problem][heuristic] = {}
                tuning_path = root_df["artifact_uri"].iloc[0].split("suprb-experimentation/")[-1]
                tuning_path = os.path.join(tuning_path, "param_history.json")
                if os.path.exists(tuning_path):
                    with open(tuning_path, "r") as f:
                        param_history = json.load(f)
                        if len(param_history.keys()) > 0:
                            n_calls = len(param_history[list(param_history.keys())[0]])
                            tuning_info[problem][heuristic]["$n_{trials}$"] = n_calls

            if fold_df is not None:
                counter += 1
                name_col = [renamed_heuristic] * fold_df.shape[0]
                current_res = fold_df.assign(Used_Representation=name_col)
                if first:
                    first = False
                    res_var = current_res
                else:
                    res_var = pd.concat([res_var, current_res])
                process_artifact_paths(current_res, renamed_heuristic,
                                       train_pareto_fronts, train_pareto_solutions,
                                       test_pareto_fronts, test_pareto_solutions)

        if counter == 0:
            print(f"No data for problem {problem}, skipping.")
            continue

        # Convert collected solution lists into arrays
        for key in train_pareto_solutions.keys():
            train_pareto_solutions[key] = np.array(train_pareto_solutions[key])
            test_pareto_solutions[key] = np.array(test_pareto_solutions[key])

        # MOO heuristics now are exactly the display names of config['heuristics']
        moo_heuristics = list(config['heuristics'].values())

        # Prepare reference (SOO) overlays
        test_ref_averages, test_ref_std = prepare_reference_stats(test_pareto_solutions, config)
        train_ref_averages, train_ref_std = prepare_reference_stats(train_pareto_solutions, config)

        n_algs = len(moo_heuristics)
        n_cols, n_rows = determine_layout(n_algs)
        dataset_key = datasets_map[problem]
        dataset_title = config['datasets'][problem]

        # Plots
        plot_hexbin(moo_heuristics, test_pareto_solutions, test_ref_averages, test_ref_std,
                    n_cols, n_rows, dataset_title, final_output_dir, dataset_key, plot_type="test")
        plot_hexbin(moo_heuristics, train_pareto_solutions, train_ref_averages, train_ref_std,
                    n_cols, n_rows, dataset_title, final_output_dir, dataset_key, plot_type="train")
        plot_mean_shift_arrows(moo_heuristics, train_pareto_solutions, test_pareto_solutions, n_cols, n_rows,
                               dataset_title, final_output_dir, dataset_key,n_bins=25)
        plot_hist(moo_heuristics, train_pareto_fronts, n_cols, n_rows, dataset_title, final_output_dir, dataset_key)
        plot_iterations_hv(res_var, moo_heuristics, final_output_dir, dataset_key, dataset_title)

        moo_heuristics = list(config['heuristics'].values())

        # Metrics (filtered to only MOO heuristics in plots)
        spread_df = compute_metric_dataframe(train_pareto_fronts, metric_spread, "Spread")
        plot_violin_metric(spread_df, config, problem, final_output_dir, "Spread", allowed_algos=moo_heuristics)

        hv_df = compute_metric_dataframe(test_pareto_fronts, metric_hypervolume, "Test Hypervolume",
                                         reference_point=np.array([1.0, 1.0]))
        plot_violin_metric(hv_df, config, problem, final_output_dir, "Test Hypervolume",
                                 allowed_algos=moo_heuristics)

        if len(config["reference_heuristics"]) > 0:
            reference_heuristic = config["reference_heuristics"][list(config["reference_heuristics"].keys())[0]]
            ga_moo_distance_df = compute_ga_comparison_dataframe(test_pareto_fronts, reference_heuristic)


            plot_violin_metric(ga_moo_distance_df, config, problem, final_output_dir,
                                     f"$f_1$ Pareto sacrifice to {reference_heuristic}",
                                     allowed_algos=moo_heuristics)

            # ---- Collect NaN percentage for Pareto Sacrifice undefined table ----
            sacrifice_col = f"$f_1$ Pareto sacrifice to {reference_heuristic}"
            per_algo_nan = compute_nan_percentage_per_algo(
                ga_moo_distance_df[ga_moo_distance_df["Used_Representation"].isin(moo_heuristics)],
                sacrifice_col,
                moo_heuristics
            )
            pareto_sacrifice_nan_stats[problem] = per_algo_nan

    # After all datasets processed: emit LaTeX table
    if len(pareto_sacrifice_nan_stats) > 0:
        generate_pareto_sacrifice_undefined_table(pareto_sacrifice_nan_stats, config, moo_heuristics, final_output_dir)
    generate_tuning_tables(tuning_info, config, final_output_dir)

if __name__ == '__main__':
    create_plots()