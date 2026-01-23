import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.chdir(
    r"C:/Users/ritch/Documents/DOE Papers/Small Sample ML Code/Python/"
)

from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd
from Jfunctions import loaders, sim_functions, util_functions

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from openml import datasets



load_id_list = [722, 734, 735, 752, 761, 821, 833, 846, 847, 976, 977, 979, 1019, 1120, 1471]
load_id_list = [722, 735, 846, 976]

######## SVC Notes #############
# works for 722, 976
# doesn't for 846, 735
# 735 works with smaller gamma range close to 0

######## Boosting notes #########
# works for 735, 722, 976, 846

nsim = 25
gridSize = 25
n_samps_list = [10, 20, 50, 100, 200]
searchMethod = "Grid"

modelLoader = loaders.model_loader_maker(
    model_constructor=lambda **kw: SVC(probability=True, **kw),
    hyperparameters=["C", "gamma"],
    hyperparameter_low=[2**-5, 2**-15],
    hyperparameter_high=[2**15, 2**3],
    logList=[True, True],
    round_param=[False, False]
)
# for 735
# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: SVC(probability=True, **kw),
#     hyperparameters=["C", "gamma"],
#     hyperparameter_low=[2**-5, 10**-15],
#     hyperparameter_high=[1, 1],
#     logList=[True, True],
#     round_param=[False, False]
# )
# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: NuSVC(probability=True, **kw),
#     hyperparameters=["gamma", "nu"],
#     hyperparameter_low=[2**-15, 2**-15],
#     hyperparameter_high=[2**3, 1 - 2**-15],
#     logList=[True, False],
#     round_param=[False, False]
# )
# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: GradientBoostingClassifier(**kw),
#     hyperparameters=["learning_rate", "n_estimators", "max_depth"],
#     hyperparameter_low=[2**-10, 100, 1],
#     hyperparameter_high=[1, 1000, 15],
#     logList=[True, True, False],
#     round_param=[False, True, True]
# )

designLoader = loaders.design_loader_maker(
    criterion="space",
    N=gridSize,
    k=len(modelLoader["hyperparameters"]),
    runDepth=500
)


def run_single_sim(j, df, n_samps, modelLoader, designLoader, gridSize, searchMethod):

    dataLoader = loaders.data_loader_maker(
        data=df,
        feature_cols=range(df.shape[1] - 1),
        target_col=df.shape[1] - 1,
        n_samps=n_samps
    )

    result = sim_functions.single_sim_sklearn_tuned(
        dataLoader,
        modelLoader,
        designLoader,
        grid_search_size=gridSize,
        searchMethod=searchMethod
    )

    return result

full_sim = []

for i, load_id in enumerate(load_id_list):

    dataset = datasets.get_dataset(load_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    df = pd.concat([X, y], axis=1)
    for _, n_samps in enumerate(n_samps_list):

        sim_hold = Parallel(n_jobs=-1)(
            delayed(run_single_sim)(
                j,
                df,
                n_samps,
                modelLoader,
                designLoader,
                gridSize,
                searchMethod
            )
            for j in tqdm(range(nsim))
        )

        sim_hold = pd.DataFrame(sim_hold)

        sim_hold["n_samps"] = n_samps
        sim_hold["model"] = str(modelLoader["model_constructor"]())
        sim_hold["Id"] = load_id
        sim_hold["n_sim"] = nsim

        full_sim.append(sim_hold)



sim_out = pd.concat(full_sim, ignore_index=True)
print()
print(np.mean(sim_out.iloc[:, 0] - sim_out.iloc[:, 1]))
print()
print(np.mean(sim_out.iloc[:, 2] - sim_out.iloc[:, 3]))
print()
print(np.mean(sim_out.iloc[:, 4] - sim_out.iloc[:, 5]))
print()
print(np.mean(sim_out.iloc[:, 1]))
print()
print(np.mean(sim_out.iloc[:, 3]))
print()
print(np.mean(sim_out.iloc[:, 5]))
sim_out.to_csv(
    f"C:/Users/ritch/Documents/DOE Papers/Small Sample ML Code/small_sample_bayes/"
    f"{modelLoader['model_constructor']()}_brier_bayes_sim_out.csv",
    index=False
)
