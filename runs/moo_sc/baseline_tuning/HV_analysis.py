import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

from matplotlib import pyplot as plt

from suprb import SupRB
from suprb.optimizer.solution.nsga2 import NonDominatedSortingGeneticAlgorithm2
from suprb.optimizer.solution.nsga2.mutation import BitFlips
from suprb.optimizer.solution.nsga2.crossover import NPoint
from suprb.optimizer.solution.nsga2.crossover import Uniform
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.logging.multi_objective import MOLogger
from problems import scale_X_y

import time

import matplotlib as mpl

# Set up plot style similar to other scripts in viz directory
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}"
})

datasets = {
    'parkinson_total': "PT",
    'combined_cycle_power_plant': "CCPP",
    'concrete_strength': "CS",
    'airfoil_self_noise': "ASN",
    'protein_structure': "PPPTS"
}

baseline_tuning_parameters = {
    "rd_sigma_mutation": {
        "combined_cycle_power_plant": 1.8979,
        "airfoil_self_noise": 2.1987,
        "concrete_strength": 2.7596,
        "protein_structure": 2.3731,
        "parkinson_total": 4.2392
    },
    "alpha_init": {
        "combined_cycle_power_plant": 0.0610,
        "airfoil_self_noise": 0.0246,
        "concrete_strength": 0.0330,
        "protein_structure": 0.0261,
        "parkinson_total": 0.0101
    },
    "Crossover_Operator": {
        "combined_cycle_power_plant": "Uniform",
        "airfoil_self_noise": "NPoint",
        "concrete_strength": "NPoint",
        "protein_structure": "Uniform",
        "parkinson_total": "Uniform"
    },
    "n_crossover": {
        "combined_cycle_power_plant": None,
        "airfoil_self_noise": 1,
        "concrete_strength": 1,
        "protein_structure": None,
        "parkinson_total": None
    },
    "sc_sigma_mutation": {
        "combined_cycle_power_plant": 0.0052,
        "airfoil_self_noise": 0.0103,
        "concrete_strength": 0.0079,
        "protein_structure": 0.0073,
        "parkinson_total": 0.0153
    }
}

def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)

if __name__ == "__main__":
    random_state = 42
    suprb_iter = 32
    sc_iter = 32

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(10 * cm, 10 * cm))

    for dataset_name, dataset_abbr in datasets.items():
        print(f"Running analysis for {dataset_abbr} ({dataset_name})")

        co = baseline_tuning_parameters["Crossover_Operator"][dataset_name]
        n_co = baseline_tuning_parameters["n_crossover"][dataset_name]
        co = NPoint(n_co) if co == "NPoint" else Uniform()
        sc_mut = baseline_tuning_parameters["sc_sigma_mutation"][dataset_name]
        alpha_init = baseline_tuning_parameters["alpha_init"][dataset_name]
        rd_mut = baseline_tuning_parameters["rd_sigma_mutation"][dataset_name]

        nsga2 = NonDominatedSortingGeneticAlgorithm2(
            n_iter=sc_iter,
            population_size=32,
            mutation=BitFlips(mutation_rate=sc_mut),
            crossover=co,
            warm_start=True,
        )

        data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
        data = data.to_numpy()

        X, y = load_dataset(name="airfoil_self_noise", return_X_y=True)
        X, y = scale_X_y(X, y)
        X, y = shuffle(X, y, random_state=random_state)

        model = SupRB(
            n_iter=suprb_iter,
            rule_discovery=ES1xLambda(n_iter=1000, mutation=HalfnormIncrease(sigma=rd_mut)),
            solution_composition=nsga2,
            logger=MOLogger(),
            random_state=random_state,
        )
        start_time = time.time()
        scores = cross_validate(
            model,
            X,
            y,
            cv=8,
            n_jobs=8,
            verbose=10,
            scoring=["r2", "neg_mean_squared_error"],
            return_estimator=True,
            fit_params={"cleanup": True},
        )
        end_time = time.time()
        logger = scores["estimator"][0].logger_
        print("Finished!")

        print(f"Time taken: {end_time - start_time:.2f} seconds")
        axes, plots = plt.subplots()
        ##### Plot Pareto Fronts #####
        pareto_front = logger.pareto_fronts_
        pareto_front = np.array(pareto_front[suprb_iter - 1])
        hvs = logger.metrics_["hypervolume"]
        hvs = np.array(list(logger.metrics_["hypervolume"].values()))

        hv = hvs[suprb_iter - 1]
        spreads = logger.metrics_["spread"]
        spread = spreads[suprb_iter - 1]

        ax.plot(hvs, label=dataset_abbr)

    ax.set_ylabel("Hypervolume")
    ax.set_xlabel("SupRB Iteration")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fige.savefig("./diss-graphs/graphs/misc/hv_it.pdf")
