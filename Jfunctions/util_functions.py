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
