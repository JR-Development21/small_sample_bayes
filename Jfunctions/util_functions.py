import numpy as np
import pandas as pd
from typing import Sequence

def design_rescaler(vec, minv, maxv):
    vec = np.asarray(vec, dtype=float)
    return ((vec + 1.0) / 2.0) * (maxv - minv) + minv


def design_scaler(vec, minv, maxv):
    vec = np.asarray(vec, dtype=float)
    return (vec - minv) / (maxv - minv) * 2.0 - 1.0

def design_log_rescaler(vec, minv, maxv, is_log: bool):
    vec = np.asarray(vec, dtype=float)

    if is_log:
        positive_vector = (vec + 1.0) / 2.0
        return 10 ** (np.log10(minv) +
                      positive_vector * (np.log10(maxv) - np.log10(minv)))
    else:
        return design_rescaler(vec, minv, maxv)


def design_log_scaler(vec, minv, maxv, is_log: bool):
    vec = np.asarray(vec, dtype=float)

    if is_log:
        positive_vector = (np.log10(vec) - np.log10(minv)) / (
            np.log10(maxv) - np.log10(minv)
        )
        return 2.0 * positive_vector - 1.0
    else:
        return design_scaler(vec, minv, maxv)

def make_namedtuple(hp_names: Sequence[str], vals):
    return dict(zip(hp_names, vals))

def string_to_dataframe(s: str) -> pd.DataFrame:
    # remove outer brackets
    inner = s.strip().lstrip("[").rstrip("]")

    # split rows by semicolon
    rows = inner.split(";")

    # parse numbers
    mat = [list(map(float, r.strip().split())) for r in rows]

    return pd.DataFrame(mat)

def prob_sq_error_loss(yhat, y):
    """
    yhat : array-like of probabilities for positive class
    y    : array-like of labels (0/1 or False/True)
    """
    yhat = np.asarray(yhat, dtype=float)
    y_bin = np.asarray(y, dtype=float)

    return np.mean((yhat - y_bin) ** 2)

def prob_log_loss(yhat, y, eps=1e-15):
    """
    Binary cross-entropy loss
    """
    yhat = np.asarray(yhat, dtype=float)
    y_bin = np.asarray(y, dtype=float)

    yhat = np.clip(yhat, eps, 1.0 - eps)

    return -np.sum(
        y_bin * np.log(yhat) +
        (1.0 - y_bin) * np.log(1.0 - yhat)
    )

def get_expected_minimum(fitted_opt, n_samples=100000):
    """
    Extracts the GP model and finds the parameters that minimize the 
    GP mean function (The Expected Minimum).
    """
    # 1. Extract the result object from the optimizer
    # optimizer_results_ is a list; usually we want the first/only run.
    result = fitted_opt.optimizer_results_[0]
    
    # 2. Get the final fitted GP model (Surrogate)
    # The model is trained on the *transformed* space (encoded categoricals)
    gp_model = result.models[-1]
    
    # 3. Create a dense grid of random samples from the search space
    # result.space handles the complexity of categorical/integer generation
    potential_params = result.space.rvs(n_samples=n_samples)
    
    # 4. Transform these parameters into the format the GP expects
    # (e.g., converting 'rbf' to integer 1, scaling log-uniforms, etc.)
    X_transformed = result.space.transform(potential_params)
    
    # 5. Predict the mean validation score for all samples
    # Note: GP predicts negative scores if maximizing, but skopt minimizes internally.
    # We simply look for the mathematical minimum of the surrogate output.
    pred_means = gp_model.predict(X_transformed)
    
    # 6. Find the index of the minimum mean
    best_idx = np.argmin(pred_means)
    
    # 7. Return the human-readable parameters
    return dict(zip(fitted_opt.search_spaces.keys(), potential_params[best_idx]))
