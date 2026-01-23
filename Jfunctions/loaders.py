import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

#still need to make exact split

def exact_balanced_split(y, n_each_class, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y)
    inds_train = []

    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) < n_each_class:
            raise ValueError(
                f"Not enough samples of class {c} to take {n_each_class}"
            )
        rng.shuffle(idx_c)
        inds_train.extend(idx_c[:n_each_class])

    inds_train = np.array(inds_train)
    inds_test = np.setdiff1d(np.arange(len(y)), inds_train)

    return inds_train, inds_test


def data_loader_maker(*, data: pd.DataFrame,
                      feature_cols: range,
                      target_col: int,
                      n_samps: int):

    y = data.iloc[:, target_col]
    train_inds, test_inds = exact_balanced_split(y, n_samps // 2)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(data.iloc[train_inds].drop(data.columns[target_col], axis=1)))
    train_scaled = pd.concat([X_train_scaled, y[train_inds].reset_index(drop=True)], axis=1)

    X_test_scaled = pd.DataFrame(scaler.transform(data.iloc[test_inds].drop(data.columns[target_col], axis=1)))
    test_scaled = pd.concat([X_test_scaled, y[test_inds].reset_index(drop=True)], axis=1)


    return {
        "train_data": train_scaled,
        "test_data": test_scaled,
        "feature_cols": feature_cols,
        "target_col": target_col
    }

def model_loader_maker(model_constructor,
                       hyperparameters,
                       hyperparameter_low,
                       hyperparameter_high,
                       round_param=None,
                       logList=None):

    return {
        "model_constructor": model_constructor,
        "hyperparameters": hyperparameters,
        "hyperparameter_low": hyperparameter_low,
        "hyperparameter_high": hyperparameter_high,
        "round_param": round_param,
        "logList": logList
    }

def design_loader_maker(*,
                        criterion="I",
                        N: int,
                        k: int,
                        runDepth: int):

    if criterion == "space":
        file_name = (
            f"C:/Users/ritch/Documents/DOE Papers/Small Sample ML Code/"
            f"smallSample.jl/designs/design_k{k}_N{N}.csv"
        )

        if os.path.isfile(file_name):
            return pd.read_csv(file_name)

        raise NotImplementedError(
            "Differential evolution design generation "
            "must be reimplemented in Python."
        )

    path = f"{csv_root}/{criterion}_sim.csv"
    df = pd.read_csv(path)

    df = df[(df["N"] == N) & (df["k"] == k)]
    rowmin = df["fgbest"].idxmin()

    design = string_to_dataframe(df.loc[rowmin, "gbest"])
    return design



