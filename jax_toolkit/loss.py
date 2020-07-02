from typing import Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp


@jax.jit
def log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/_classification.py#L2123 """
    # If single dimension, assume binary clasification problem
    if y_true.ndim == 1:
        y_true = y_true[:, jnp.newaxis]
        y_pred = y_pred[:, jnp.newaxis]
    if y_true.shape[1] == 1:
        y_true = jnp.append(1 - y_true, y_true, axis=1)
        y_pred = jnp.append(1 - y_pred, y_pred, axis=1)

    # Clipping
    eps = 1e-15
    y_pred = y_pred.astype(jnp.float32).clip(eps, 1 - eps)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, jnp.newaxis]
    loss = (y_true * -jnp.log(y_pred)).sum(axis=1)
    mean_loss_per_sample = jnp.average(loss)
    return mean_loss_per_sample


@jax.jit
def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = jnp.average(jnp.abs(y_pred - y_true), axis=0)
    mean_loss_per_sample = jnp.average(loss)
    return mean_loss_per_sample


@jax.jit
def median_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = jnp.median(jnp.abs(y_pred - y_true), axis=0)
    mean_loss_per_sample = jnp.average(loss)
    return mean_loss_per_sample


@jax.jit
def max_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = jnp.abs(y_pred - y_true).max(axis=0)
    mean_loss_per_sample = jnp.average(loss)
    return mean_loss_per_sample


@jax.jit
def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.average((y_true - y_pred) ** 2)


@jax.jit
def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/_regression.py#L513 """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    denominator = ((y_true - jnp.average(y_true, axis=0)) ** 2).sum(axis=0)
    # if denominator.sum() == 0:
    #     # if numerator.sum() == 0:  # i.e. all numerator is 0, so perfect fit
    #     return jnp.array(1)
        # else:
        #     return jnp.array(0)  # i.e. constant y
    r2_scores = 1 - (numerator / denominator)[jnp.nonzero(denominator)]
    # handle nan resulting from division by 0
    if y_true.ndim == 1:  # i.e. multi-output
        return r2_scores
    else:
        return jnp.average(r2_scores[~jnp.isnan(r2_scores)])


# =====
# UTILS
# =====
SUPPORTED_LOSSES: Dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "log_loss": log_loss,
    "mean_absolute_error": mean_absolute_error,
    "median_absolute_error": median_absolute_error,
    "max_absolute_error": max_absolute_error,
    "mean_squared_error": mean_squared_error,
    "r2_score": r2_score,

}


def get_loss_function(
    net_transform: hk.Transformed, loss: str
) -> Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    try:
        loss_function = SUPPORTED_LOSSES[loss]

        @jax.jit
        def loss_function_wrapper(params: hk.Params, x: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
            y_pred: jnp.ndarray = net_transform.apply(params, x)
            loss_value: jnp.ndarray = loss_function(y_true, y_pred)
            return loss_value

        return loss_function_wrapper  # type: ignore
    except KeyError:
        raise LossNotCurrentlySupportedException(loss)


class LossNotCurrentlySupportedException(Exception):
    def __init__(self, loss: str):
        super().__init__(
            f"Loss={loss} is not currently supported. Currently supported losses are: {list(SUPPORTED_LOSSES.keys())}"
        )
