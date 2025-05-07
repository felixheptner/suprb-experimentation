import sys

import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.multi_objective import MOLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import nsga2, nsga3, spea2
from suprb.optimizer.rule import es, origin, mutation
from suprb.optimizer.solution.sampler import BetaSolutionSampler, DiversitySolutionSampler


random_state = 42

opt_dict = {"nsag2": nsga2.NonDominatedSortingGeneticAlgorithm2,
            "nsga3": nsga3.NonDominatedSortingGeneticAlgorithm3,
            "spea2": spea2.StrengthParetoEvolutionaryAlgorithm2}


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
@click.option('-o', '--optimizer', type=click.STRING, default='nsga3')
@click.option('-s', '--sampler', type=click.STRING, default='beta')

def run(problem: str, job_id: str, optimizer: str, sampler: str):
    print(f"Problem is {problem}, with job id {job_id} and optimizer {optimizer}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    sampler_dict = {
        "uniform": BetaSolutionSampler(a=1.0, b=1.0, projected=False),
        "beta": BetaSolutionSampler(a=1.5, b=1.5, projected=False),
        "beta_projection": BetaSolutionSampler(a=1.5, b=1.5, projected=True),
        "diversity": DiversitySolutionSampler()
    }

    estimator = SupRB(
        rule_discovery=es.ES1xLambda(),
        solution_composition=opt_dict[optimizer](n_iter=32, population_size=32, sampler=sampler_dict[sampler]),
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger(
            [('stdout', StdoutLogger()), ('default', MOLogger())]),
    )

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=1000,
        timeout=60*60*24*3 if not sys.gettrace() else 60,
        scoring='hypervolume',
        verbose=10
    )

    @param_space()
    def suprb_ES_MOO_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, np.sqrt(X.shape[1])]

        params.rule_discovery__mutation__sigma = trial.suggest_float('rule_discovery__mutation__sigma', *sigma_space)
        params.rule_discovery__init__fitness__alpha = trial.suggest_float(
            'rule_discovery__init__fitness__alpha', 0.01, 0.2)

        # SC
        params.solution_composition__crossover = trial.suggest_categorical(
            'solution_composition__crossover', ['NPoint', 'Uniform'])
        params.solution_composition__crossover = getattr(nsga3.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, nsga3.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float(
            'solution_composition__mutation_rate', 0, 0.1)

        # Sampler
        if sampler in ("beta", "beta_projection"):
            params.solution_composition__sampler__a = trial.suggest_float(
                'solution_composition__sampler__a', 0.1, 10)
            params.solution_composition__sampler__b = trial.suggest_float(
                'solution_composition__sampler__b', 0.1, 10)


    experiment_name = (f'SampComp {optimizer} s:{sampler} j:{job_id} p:{problem}')
    print(experiment_name)
    experiment = Experiment(name=experiment_name,  verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_ES_MOO_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
