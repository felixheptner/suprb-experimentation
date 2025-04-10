import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.linear_model import Ridge
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
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import rs, origin, mutation
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    else:
        enc = LabelEncoder()
        dataset = fetch_openml(name=name, version=1)

        if name == "meta":
            dataset.data.DS_Name = enc.fit_transform(dataset.data.DS_Name)
            dataset.data.Alg_Name = enc.fit_transform(dataset.data.Alg_Name)
            dataset.data = dataset.data.drop(
                dataset.data.columns[dataset.data.isna().any()].tolist(), axis=1)

        if name == "chscase_foot":
            dataset.data.col_1 = enc.fit_transform(dataset.data.col_1)

        if isinstance(dataset.data, np.ndarray):
            X = dataset.data
        elif isinstance(dataset.data, pd.DataFrame) or isinstance(dataset.data, pd.Series):
            X = dataset.data.to_numpy(dtype=float)
        else:
            X = dataset.data.toarray()

        if isinstance(dataset.target, np.ndarray):
            y = dataset.target
        elif isinstance(dataset.target, pd.DataFrame) or isinstance(dataset.target, pd.Series):
            y = dataset.target.to_numpy(dtype=float)
        else:
            y = dataset.target.toarray()


        return X, y


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_discovery=rs.RandomSearch(
            init=rule.initialization.HalfnormInit(fitness=rule.fitness.VolumeWu(),
                                                  model=Ridge(alpha=0.01,
                                                  random_state=random_state)),
            origin_generation=origin.SquaredError(),
        ),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32),
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger(
            [('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=10_000,
        timeout=72 * 60 * 60,  # 72 hours
        scoring='fitness',
        verbose=10
    )

    @param_space()
    def suprb_RS_GA_space(trial: Trial, params: Bunch):
        # RS
        params.rule_discovery__rules_generated = trial.suggest_int(
            'rule_discovery__rules_generated', 5, 500)
        params.rule_discovery__init__sigma = trial.suggest_float(
                 'rule_discovery__init__sigma', 0.01, 1)

        # GA
        params.solution_composition__selection = trial.suggest_categorical(
            'solution_composition__selection', ['RouletteWheel', 'Tournament',
                                                'LinearRank', 'Random'])
        params.solution_composition__selection = getattr(ga.selection, params.solution_composition__selection)()

        if isinstance(params.solution_composition__selection, ga.selection.Tournament):
            params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)

        params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover',
                                                     ['NPoint', 'Uniform'])
        params.solution_composition__crossover = getattr(ga.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float('solution_composition__mutation_rate', 0,
                                                             0.1)

    experiment = Experiment(name=f'{problem} RS Tuning & Experimentation',
                            verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_RS_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment("RS Tuning & Experimentation")
    log_experiment(experiment)


if __name__ == '__main__':
    run()


