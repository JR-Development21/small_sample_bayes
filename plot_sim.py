import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.chdir(
    r"C:/Users/ritch/Documents/DOE Papers/Small Sample ML Code/Python/"
)

from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from Jfunctions import loaders, sim_functions, util_functions

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from openml import datasets



load_id_list = [976]

######## SVC Notes #############
# works for 722, 976
# doesn't for 846, 735
# 735 works with smaller gamma range close to 0

######## Boosting notes #########
# works for 735, 722, 976, 846

nsim = 10
gridSize = 27
n_samps_list = [10]
searchMethod = "Grid"

# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: SVC(probability=True, **kw),
#     hyperparameters=["C", "gamma"],
#     hyperparameter_low=[2**-5, 2**-15],
#     hyperparameter_high=[2**15, 2**3],
#     logList=[True, True],
#     round_param=[False, False]
# )
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
#     model_constructor=lambda **kw: SVC(kernel="linear", probability=True, **kw),
#     hyperparameters=["C"],
#     hyperparameter_low=[2**-5],
#     hyperparameter_high=[2**15],
#     logList=[True],
#     round_param=[False]
# )
modelLoader = loaders.model_loader_maker(
    model_constructor=lambda **kw: GradientBoostingClassifier(**kw),
    hyperparameters=["learning_rate"],
    hyperparameter_low=[2**-10],
    hyperparameter_high=[1],
    logList=[True],
    round_param=[False]
)
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


def run_single_sim(j, df, n_samps, modelLoader, designLoader):

    dataLoader = loaders.data_loader_maker(
        data=df,
        feature_cols=range(df.shape[1] - 1),
        target_col=df.shape[1] - 1,
        n_samps=n_samps
    )

    result = sim_functions.plot_sim(
        dataLoader,
        modelLoader,
        designLoader
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
                designLoader
            )
            for j in tqdm(range(nsim))
        )

        sim_hold = pd.DataFrame(sim_hold)

        sim_hold["n_samps"] = n_samps
        sim_hold["model"] = str(modelLoader["model_constructor"]())
        sim_hold["Id"] = load_id
        sim_hold["n_sim"] = nsim

        full_sim.append(sim_hold)



quad_list = full_sim[0]['quad_model']
cubic_list = full_sim[0]['cubic_model']
df_list = full_sim[0]['df_surrogate']

def plot_quad_cubic_from_models(
    quad_models,
    cubic_models,
    dfs,
    x_col,
    y_col,
    n_grid=200
):
    """
    quad_models : list of statsmodels results (quadratic)
    cubic_models: list of statsmodels results (cubic)
    dfs         : list of DataFrames
    x_col       : str, predictor column name
    y_col       : str, response column name
    """

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=False, sharey=False)
    axes = axes.ravel()

    for i in range(len(dfs)):
        ax = axes[i]
        df = dfs[i]

        # ---- scatter ----
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=30)

        # ---- prediction grid ----
        x_min, x_max = df[x_col].min(), df[x_col].max()
        x_grid = np.linspace(x_min, x_max, n_grid)
        pred_df = pd.DataFrame({x_col: x_grid})

        # ---- quadratic prediction ----
        y_quad = quad_models[i].predict(pred_df)
        ax.plot(x_grid, y_quad, linewidth=2, label="Quadratic")

        # ---- cubic prediction ----
        y_cubic = cubic_models[i].predict(pred_df)
        ax.plot(x_grid, y_cubic, linewidth=2, linestyle="--", label="Cubic")

        ax.set_title(f"Sim {i+1}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_quad_cubic_from_models(
    quad_models=quad_list,
    cubic_models=cubic_list,
    dfs=df_list,
    x_col="learning_rate",
    y_col="Accuracy"
)
