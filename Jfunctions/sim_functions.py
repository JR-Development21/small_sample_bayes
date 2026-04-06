import numpy as np
import pandas as pd

from Jfunctions import util_functions

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    RepeatedStratifiedKFold
)
from sklearn.metrics import balanced_accuracy_score, log_loss, brier_score_loss, make_scorer
from sklearn.base import clone
from typing import Callable, List, Dict, Any, Optional


neg_brier_scorer = make_scorer(
    brier_score_loss,
    greater_is_better=False,
    response_method="predict_proba",
    pos_label='P' # Optionally specify the positive label
)

def repeated_bbc_cv(
    X: np.ndarray,
    y: np.ndarray,
    Xtest: np.ndarray,
    ytest: np.ndarray,
    estimator,
    param_list: List[dict],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_bootstraps: int = 1000,
    random_state: Optional[int] = None,
    use_predict_proba_if_available: bool = True,
    ci_alpha: float = 0.05
) -> Dict[str, Any]:

    rng = np.random.default_rng(random_state)

    if use_predict_proba_if_available:
        y = y
        ytest = ytest
    else:
        y = pd.Series([1 if i=="P" else 0 for i in y])
        ytest = pd.Series([1 if i=="P" else 0 for i in ytest])

    n = len(y)
    C = len(param_list)

    # ---- pooled OOF predictions ----
    Pi = np.zeros((n * n_repeats, C), dtype=float)
    Pi_base = np.zeros((n * n_repeats, 1), dtype=float)

    # must match repeat-major ordering
    y_rep = np.tile(y, n_repeats)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    # ---- collect OOF predictions ----
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        r = fold_id // n_splits
        row_idx = test_idx + r * n

        # fit base model
        model = clone(estimator)
        model.fit(X.iloc[train_idx, :], y[train_idx])
        if use_predict_proba_if_available and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X.iloc[test_idx, :])[:, 1]
        else:
            preds = model.predict(X.iloc[test_idx, :])

        Pi_base[row_idx, 0] = preds

        # fit other parameter combinations
        for c, params in enumerate(param_list):

            model = clone(estimator)
            model.set_params(**params)

            model.fit(X.iloc[train_idx, :], y[train_idx])

            if use_predict_proba_if_available and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X.iloc[test_idx, :])[:, 1]
            else:
                preds = model.predict(X.iloc[test_idx, :])

            Pi[row_idx, c] = preds

    # ---- select best config on full pooled predictions (CVT step) ----
    full_scores = []
    for c in range(C):
        try:
            s = metric_fn(y_rep, Pi[:, c])
        except Exception:
            s = -np.inf
        full_scores.append(s)

    best_config_full = int(np.argmin(full_scores))

    # ---- BBC bootstrap ----
    L = []
    L_rand = []
    L_base = []

    for _ in range(n_bootstraps):

        # sample instances
        sampled = rng.integers(0, n, size=n)

        # find OOB instances
        present = np.zeros(n, dtype=bool)
        present[np.unique(sampled)] = True
        oob = np.where(~present)[0]

        if len(oob) == 0:
            continue

        # expand rows (include all repeats)
        boot_rows = np.concatenate([sampled + r*n for r in range(n_repeats)])
        oob_rows  = np.concatenate([oob     + r*n for r in range(n_repeats)])

        # ---- selection inside bootstrap ----
        scores_boot = []
        for c in range(C):
            try:
                s = metric_fn(y_rep[boot_rows], Pi[boot_rows, c])
            except Exception:
                s = np.inf
            scores_boot.append(s)

        best_c = int(np.argmin(scores_boot))
        rand_c =  rng.integers(0, C, size=1)

        # ---- evaluate selected config on OOB ----
        try:
            Lb = metric_fn(y_rep[oob_rows], Pi[oob_rows, best_c])
            Lb_rand = metric_fn(y_rep[oob_rows], Pi[oob_rows, rand_c])
            Lb_base = metric_fn(y_rep[oob_rows], Pi_base[oob_rows, 0])
        except Exception:
            continue

        L.append(Lb)
        L_rand.append(Lb_rand)
        L_base.append(Lb_base)

    L = np.array(L)
    L_rand = np.array(L_rand)
    L_base = np.array(L_base)

    # ---- summarize ----
    mean_perf = np.mean(L)
    ci = np.percentile(L, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])

    mean_rand = np.mean(L_rand)
    ci_rand = np.percentile(L_rand, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])
    ci_diff_rand = np.percentile(L - L_rand, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])

    mean_base = np.mean(L_base)
    ci_base = np.percentile(L_base, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])
    ci_diff_base = np.percentile(L - L_base, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])


    # Get back best models
    model = clone(estimator)
    model.set_params(**param_list[best_config_full])

    model.fit(X, y)
    if use_predict_proba_if_available and hasattr(model, "predict_proba"):
        yhat_test = model.predict_proba(Xtest)[:, 1]
    else:
        yhat_test = model.predict(Xtest)

    bbc_test_func = metric_fn(ytest, yhat_test)

    #define base model
    base_model = clone(estimator)

    base_model.fit(X, y)
    if use_predict_proba_if_available and hasattr(base_model, "predict_proba"):
        yhat_test = base_model.predict_proba(Xtest)[:, 1]
    else:
        yhat_test = base_model.predict(Xtest)
    base_test_func = metric_fn(ytest, yhat_test)

    return {
        "bbc_estimate": mean_perf,
        "bbc_ci": ci,
        "rand_estimate": mean_rand,
        "rand_ci": ci_rand,
        "diff_rand_ci": ci_diff_rand,
        "base_estimate": mean_base,
        "base_ci": ci_base,
        "diff_base_ci": ci_diff_base,
        "bootstrap_values": L,
        "rand_values": L_rand,
        "best_config_index": best_config_full,
        "best_params": param_list[best_config_full],
        "bbc_test_func": bbc_test_func,
        "base_test_func": base_test_func,
        # "bbc_test_ba": bbc_test,
        # "base_test_ba": base_test
    }



def single_sim_conf_int(
        dataLoader, 
        modelLoader,
        designLoader,
        use_predict_proba_if_available: bool = True,
        metric_fn = lambda x,y: brier_score_loss(x,y,pos_label='P')
        ):
    Xtrain = dataLoader["train_data"].iloc[:, dataLoader["feature_cols"]]
    ytrain = dataLoader["train_data"].iloc[:, dataLoader["target_col"]]
    Xtest = dataLoader["test_data"].iloc[:, dataLoader["feature_cols"]]
    ytest = dataLoader["test_data"].iloc[:, dataLoader["target_col"]]
    estimator = modelLoader["model_constructor"]()
    des_grid_real = pd.DataFrame()

    hyperparameters = modelLoader["hyperparameters"]
    n_hyp = len(hyperparameters)

    logList = modelLoader["logList"]
    if logList is None:
        logList = [False] * n_hyp

    lows = modelLoader["hyperparameter_low"]
    highs = modelLoader["hyperparameter_high"]

    for i, hp in enumerate(hyperparameters):
        des_grid_real[hp] = util_functions.design_log_rescaler(
            designLoader.iloc[:, i].values,
            lows[i],
            highs[i],
            logList[i]
        )

    des_grid = []

    for _, row in des_grid_real.iterrows():
        params = {}

        for i, hp in enumerate(modelLoader["hyperparameters"]):
            val = row[hp]

            if modelLoader["round_param"][i]:
                params[hp] = int(round(val))
            else:
                params[hp] = float(val)
        des_grid.append(params)


    output = repeated_bbc_cv(Xtrain, 
                             ytrain,
                             Xtest, 
                             ytest, 
                             estimator, 
                             des_grid, 
                             metric_fn=metric_fn, 
                             n_bootstraps = 1000, 
                             use_predict_proba_if_available = use_predict_proba_if_available)
    return output
