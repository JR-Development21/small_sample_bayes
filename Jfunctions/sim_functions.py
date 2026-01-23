import numpy as np
import pandas as pd

from Jfunctions import util_functions

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import balanced_accuracy_score, log_loss, brier_score_loss, make_scorer
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from skopt import BayesSearchCV
from skopt.space import Real, Integer

neg_brier_scorer = make_scorer(
    brier_score_loss,
    greater_is_better=False,
    response_method="predict_proba",
    pos_label='P' # Optionally specify the positive label
)

def single_sim_sklearn_tuned(
    data_loader,
    model_loader,
    design_loader,
    grid_search_size=100,
    searchMethod="Grid"
):
    hyperparameters = model_loader["hyperparameters"]
    n_hyp = len(hyperparameters)

    rep_hyp = int(np.ceil(grid_search_size ** (1 / n_hyp)))

    logList = model_loader["logList"]
    if logList is None:
        logList = [False] * n_hyp

    lows = model_loader["hyperparameter_low"]
    highs = model_loader["hyperparameter_high"]
    round_param = model_loader["round_param"]
    base_model = model_loader["model_constructor"]()
    param_grid = {}

    for i, hp in enumerate(hyperparameters):
        if logList[i]:
            param_grid[hp] = np.logspace(
                np.log10(lows[i]),
                np.log10(highs[i]),
                rep_hyp
            )
        else:
            param_grid[hp] = np.linspace(lows[i], highs[i], rep_hyp)

        if round_param[i]:
            param_grid[hp] = np.unique(np.round(param_grid[hp]).astype(int))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    if searchMethod == "Grid":
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=1
        )
    elif searchMethod == "Random":
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=grid_search_size,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=1
        )
    else:
        raise NotImplementedError("Latin Hypercube requires custom implementation")

    Xtrain = data_loader["train_data"].iloc[:, data_loader["feature_cols"]]
    ytrain = data_loader["train_data"].iloc[:, data_loader["target_col"]]

    Xtest = data_loader["test_data"].iloc[:, data_loader["feature_cols"]]
    ytest = data_loader["test_data"].iloc[:, data_loader["target_col"]]

    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_

    yhat_test = best_model.predict(Xtest)


    grid_search_test_acc = balanced_accuracy_score(ytest, yhat_test)
    grid_search_val_acc = search.best_score_

    gs_hyp_vals = {
        f"{hp}_gs": search.best_params_[hp]
        for hp in hyperparameters
    }

    bs_logs = ["log-uniform" if logtrue else "uniform" for logtrue in logList]
    bs_ranges = [Integer(low = lows[i], high = highs[i], prior = bs_logs[i]) if round_param[i] else Real(low = lows[i], high = highs[i], prior = bs_logs[i]) for i in range(n_hyp)]
    opt = BayesSearchCV(
        base_model,
        {hyperparameters[i]:bs_ranges[i] for i in range(n_hyp)},
        n_iter=grid_search_size,
        scoring="balanced_accuracy",
        cv=5
    )

    opt.fit(Xtrain, ytrain)

    bayes_val_acc = opt.best_score_
    best_model = opt.best_estimator_
    best_model.fit(Xtrain, ytrain)
    bayes_test_acc = balanced_accuracy_score(
        ytest,
        best_model.predict(Xtest)
    )

    bayes_hyp_vals = {
        f"{hp}_bayes": opt.best_params_[hp]
        for hp in hyperparameters
    }

    des_grid_real = pd.DataFrame()

    for i, hp in enumerate(hyperparameters):
        des_grid_real[hp] = util_functions.design_log_rescaler(
            design_loader.iloc[:, i].values,
            lows[i],
            highs[i],
            logList[i]
        )

    des_scores = []

    for _, row in des_grid_real.iterrows():
        params = {}

        for i, hp in enumerate(model_loader["hyperparameters"]):
            val = row[hp]

            if model_loader["round_param"][i]:
                params[hp] = int(round(val))
            else:
                params[hp] = float(val)

        model = model_loader["model_constructor"](**params)

        scores = cross_val_score(
            model,
            Xtrain,
            ytrain,
            scoring=neg_brier_scorer,
            # scoring = "neg_log_loss",
            cv=cv
        )
        des_scores.append(-scores.mean())

    df_surrogate = pd.DataFrame()

    for i, hp in enumerate(hyperparameters):
        df_surrogate[hp] = util_functions.design_log_scaler(
            des_grid_real[hp].values,
            lows[i],
            highs[i],
            logList[i]
        )

    df_surrogate["Accuracy"] = des_scores

    terms = hyperparameters

    main_terms = " + ".join(terms)
    quad_terms = " + ".join([f"I({t}**2)" for t in terms])
    cubic_terms = " + ".join([f"I({t}**3)" for t in terms])

    interaction_terms = " + ".join(
        f"{t1}:{t2}"
        for i, t1 in enumerate(terms)
        for t2 in terms[i+1:]
    )

    all_terms = " + ".join(
        t for t in [main_terms, quad_terms, cubic_terms, interaction_terms] if t
    )

    formula = f"Accuracy ~ {all_terms}"
    lm_model = smf.ols(formula, data=df_surrogate).fit()

    def f_to_min(x):
        row = {hp: x[i] for i, hp in enumerate(hyperparameters)}
        df = pd.DataFrame([row])
        return lm_model.predict(df).iloc[0]

    bounds = [(-1.0, 1.0)] * n_hyp

    start_vals = [(0)] * n_hyp

    res = minimize(
        f_to_min,
        x0=start_vals,
        bounds=bounds,
        method="L-BFGS-B"
    )

    doe_pars_coded = res.x
    doe_vals = []
    for i in range(n_hyp):
        v = util_functions.design_log_rescaler(
            [doe_pars_coded[i]],
            lows[i],
            highs[i],
            logList[i]
        )[0]
        if round_param[i]:
            v = int(round(v))
        doe_vals.append(v)

    doe_params = dict(zip(hyperparameters, doe_vals))
    doe_model = model_loader["model_constructor"](**doe_params)

    doe_model.fit(Xtrain, ytrain)
    yhat_doe = doe_model.predict(Xtest)

    doe_test_acc = balanced_accuracy_score(ytest, yhat_doe)
    doe_val_acc = cross_val_score(
        doe_model,
        Xtrain,
        ytrain,
        scoring="balanced_accuracy",
        cv=cv
    ).mean()

    baseline = model_loader["model_constructor"]()
    baseline.fit(Xtrain, ytrain)

    yhat_baseline = baseline.predict(Xtest)
    baseline_test_acc = balanced_accuracy_score(ytest, yhat_baseline)

    baseline_val_acc = cross_val_score(
        baseline,
        Xtrain,
        ytrain,
        scoring="balanced_accuracy",
        cv=cv
    ).mean()

    return {
        "grid_search_val_acc": grid_search_val_acc,
        "grid_search_test_acc": grid_search_test_acc,
        "bayes_val_acc": bayes_val_acc,
        "bayes_test_acc": bayes_test_acc,
        "doe_val_acc": doe_val_acc,
        "doe_test_acc": doe_test_acc,
        "baseline_val_acc": baseline_val_acc,
        "baseline_test_acc": baseline_test_acc,
        **gs_hyp_vals,
        **bayes_hyp_vals,
        **{f"{hp}_doe": doe_params[hp] for hp in hyperparameters}
    }

def plot_sim(
    data_loader,
    model_loader,
    design_loader
):
    hyperparameters = model_loader["hyperparameters"]
    n_hyp = len(hyperparameters)


    logList = model_loader["logList"]
    if logList is None:
        logList = [False] * n_hyp

    lows = model_loader["hyperparameter_low"]
    highs = model_loader["hyperparameter_high"]
    round_param = model_loader["round_param"]
    

    Xtrain = data_loader["train_data"].iloc[:, data_loader["feature_cols"]]
    ytrain = data_loader["train_data"].iloc[:, data_loader["target_col"]]

    Xtest = data_loader["test_data"].iloc[:, data_loader["feature_cols"]]
    ytest = data_loader["test_data"].iloc[:, data_loader["target_col"]]

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    des_grid_real = pd.DataFrame()

    for i, hp in enumerate(hyperparameters):
        des_grid_real[hp] = util_functions.design_log_rescaler(
            design_loader.iloc[:, i].values,
            lows[i],
            highs[i],
            logList[i]
        )

    des_scores = []

    for _, row in des_grid_real.iterrows():
        params = {}

        for i, hp in enumerate(model_loader["hyperparameters"]):
            val = row[hp]

            if model_loader["round_param"][i]:
                params[hp] = int(round(val))
            else:
                params[hp] = float(val)

        model = model_loader["model_constructor"](**params)

        scores = cross_val_score(
            model,
            Xtrain,
            ytrain,
            # scoring=neg_brier_scorer,
            scoring = "neg_log_loss",
            cv=cv
        )
        des_scores.append(-scores.mean())

    df_surrogate = pd.DataFrame()

    for i, hp in enumerate(hyperparameters):
        df_surrogate[hp] = util_functions.design_log_scaler(
            des_grid_real[hp].values,
            lows[i],
            highs[i],
            logList[i]
        )

    df_surrogate["Accuracy"] = des_scores

    terms = hyperparameters

    main_terms = " + ".join(terms)
    quad_terms = " + ".join([f"I({t}**2)" for t in terms])
    cubic_terms = " + ".join([f"I({t}**3)" for t in terms])

    interaction_terms = " + ".join(
        f"{t1}:{t2}"
        for i, t1 in enumerate(terms)
        for t2 in terms[i+1:]
    )

    all_terms = " + ".join(
        t for t in [main_terms, quad_terms, cubic_terms, interaction_terms] if t
    )

    formula = f"Accuracy ~ {all_terms}"
    lm_model_cubic = smf.ols(formula, data=df_surrogate).fit()

    all_terms = " + ".join(
        t for t in [main_terms, quad_terms, interaction_terms] if t
    )

    formula = f"Accuracy ~ {all_terms}"
    lm_model_quadratic = smf.ols(formula, data=df_surrogate).fit()


    return {
        "cubic_model": lm_model_cubic,
        "quad_model": lm_model_quadratic,
        "df_surrogate": df_surrogate
    }

