import json
import os
import sys

import mlflow
import numpy as np
import time

from logging_output_scripts import violin_and_swarm_plots
from logging_output_scripts import moo_plots
from logging_output_scripts.stat_analysis import calvo, ttest
from logging_output_scripts.utils import filter_runs

saga_datasets = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    # "energy_cool": "Energy Efficiency Cooling",
    "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",
    "parkinson_total": "Parkinson's Telemonitoring"
}

def mlruns_to_csv(datasets, subdir, normalize):
    all_runs_df = mlflow.search_runs(search_all_experiments=True)

    print("Dataset\t\t\tMin MSE\tMax MSE\tMin Complexity\tMax Complexity")
    for dataset in datasets:
        mse = "metrics.test_neg_mean_squared_error"
        complexity = "metrics.elitist_complexity"
        hypervolume = "metrics.hypervolume"
        sc_iters = "metrics.sc_iterations"
        spread = "metrics.spread"
        df = all_runs_df[all_runs_df["tags.mlflow.runName"].str.contains(
            dataset, case=False, na=False) & (all_runs_df["tags.fold"] == 'True')]
        df = df[["tags.mlflow.runName", "artifact_uri", mse, complexity, hypervolume, spread, sc_iters]]
        print(f"{dataset}\t\t\t{np.min(df[mse]):.4f}\t{np.max(df[mse]):.4f}\t{np.min(df[complexity]):.4f}\t"
              f"{np.max(df[complexity]):.4f}")

        df[mse] *= -1
        if normalize:
            df[mse] = (df[mse] - np.min(df[mse])) / (np.max(df[mse]) - np.min(df[mse]))
            df[complexity] = (df[complexity] - np.min(df[complexity])) / (np.max(df[complexity]) - np.min(df[complexity]))
        df.to_csv(f"mlruns_csv/{subdir}/{dataset}_all.csv", index=False)


saga = {
    "s:ga": "GA",
    "s:saga1": "SAGA1",
    "s:saga2": "SAGA2",
    "s:saga3": "SAGA3",
    "s:sas": "SAGA4"
}

ga_baseline = {
    "Baseline c:ga_no_tuning": "No Tuning",
    "Baseline c:ga32": "GA 32",
    "Baseline c:ga64": "GA 64",
}

moo_baseline = {
    #"Baseline nsga2": "NSGA-II",
    "Baseline nsga3": "NSGA-III",
}

test = {
    "Test nsga3": "NSGA-III",
    "Test spea2": "SPEA2"
}

moo_sampler = {
    "SampComp nsga2 s:uniform j:733226": "Uniform",
    "nsga2 Baseline j:730083": r"Beta $\alpha = \beta = 1.5$",
    "SampComp nsga2 s:beta j:733231": "Beta Tuned",
    "SampComp nsga2 s:beta_projection j:733236": "Beta Projection",
    "SampComp nsga2 s:diversity j:733241": "Diversity",
}

moo_early = {
    "nsga2 Baseline j:730083": "NSGA-II",
    "Early Stopping nsga2": "NSGA-II ES",
    "nsga3 Baseline": "NSGA-III",
    "Early Stopping nsga3": "NSGA-III ES",
    "spea2 Baseline": "SPEA2",
    "Early Stopping spea2": "SPEA2 ES",
}

moo_ts_noes = {
    "nsga2 Baseline j:730083": "NSGA-II",
    "TSComp nsga2 c:ga-moo j": "GA - NSGA-II",
    "TSComp nsga2 c:ga_without_tuning-moo ": "GA Untuned - NSGA-II",
}

moo_ts_es = {
    "nsga2 Baseline j:730083": "NSGA-II",
    "TSComp nsga2 c:ga-moo e:True": "GA - NSGA-II",
    "TSComp nsga2 c:ga_without_tuning-moo e:True": "GA Untuned - NSGA-II",
}

pop_size = {
    "nsga2 Baseline j:730083": "N = 32",
    "ProjComp nsga2 ps:64": "N = 64",
    "ProjComp nsga2 ps:128": "N = 128"
}

def run_main():
    with open("logging_output_scripts/config.json", "r") as f:
        config = json.load(f)

    config["datasets"] = saga_datasets if setting is not test else {"parkinson_total": "Parkinson's Telemonitoring"}

    config["output_directory"] = setting[0]
    if not os.path.isdir("diss-graphs/graphs"):
        os.mkdir("diss-graphs/graphs")

    if not os.path.isdir(config["output_directory"]):
        os.mkdir(config["output_directory"])

    config["normalize_datasets"] = setting[3]

    config["heuristics"] = setting[1]
    config["data_directory"] = setting[4]

    with open("logging_output_scripts/config.json", "w") as f:
        json.dump(config, f)

    time.sleep(10)

    if config["data_directory"] == "mlruns":
        all_runs_df = mlflow.search_runs(search_all_experiments=True)
        filter_runs(all_runs_df)

    """if setting[0] == "diss-graphs/graphs/MOO":
        ttest(latex=True, cand1="Baseline nsga2", cand2="Baseline nsga3", cand1_name="NSGA-II", cand2_name="NSGA-III")
        ttest(latex=True, cand1="Baseline nsga2", cand2="Baseline spea2", cand1_name="NSGA-II", cand2_name="SPEA2")

        ttest(latex=True, cand1="Baseline nsga3", cand2="Baseline spea2", cand1_name="NSGA-III", cand2_name="SPEA2")
    """
    # calvo(ylabel=setting[2])
    moo_plots.create_plots()
    violin_and_swarm_plots.create_plots()


if __name__ == '__main__':
    ga_base = ["diss-graphs/graphs/GA_BASELINE", ga_baseline, "Solution Composition", False, "mlruns_csv/GA_BASELINE"]
    moo_algos = ["diss-graphs/graphs/MOO", moo_baseline, "Solution Composition", False, "mlruns_csv/MOO"]
    moo_sampler = ["diss-graphs/graphs/SAMPLER", moo_sampler, "Solution Composition", False, "mlruns_csv/SAMPLER"]
    moo_early = ["diss-graphs/graphs/EARLY", moo_early, "Solution Composition", False, "mlruns_csv/EARLY"]
    moo_ts_noes = ["diss-graphs/graphs/TS", moo_ts_noes, "Solution Composition", False, "mlruns_csv/TS"]
    moo_ts_es = ["diss-graphs/graphs/TSES", moo_ts_es, "Solution Composition", False, "mlruns_csv/TSES"]
    pop_size = ["diss-graphs/graphs/POP", pop_size, "Solution Composition", False, "mlruns_csv/POP"]
    test = ["diss-graphs/graphs/TEST", test, "Solution Composition", False, "mlruns_csv/TEST"]

    # setting = ga_base
    setting = test
    # ffsetting = moo_algos
    # setting = moo_sampler
    # setting = moo_early
    # setting = moo_ts_noes
    # setting = moo_ts_es
    # setting = pop_size

    mlruns_to_csv(saga_datasets if setting is not test else {"parkinson_total": "Parkinson's Telemonitoring"},
                  subdir=setting[-1].split("/")[-1], normalize=True)
    run_main()

adel = {"SupRB": "SupRB",
        "Random Forest": "RF",
        "Decision Tree": "DT", }
