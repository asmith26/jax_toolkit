from typing import Callable, Dict

import haiku as hk
import jax
from jax import numpy as jnp


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
    y_pred = jnp.clip(y_pred.astype(jnp.float32), eps, 1 - eps)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, jnp.newaxis]
    loss = (y_true * -jnp.log(y_pred)).sum(axis=1)
    mean_loss_per_sample = jnp.average(loss)
    return mean_loss_per_sample


@jax.jit
def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.average((y_true - y_pred) ** 2)  # weights=sample_weight)


# =====
# UTILS
# =====
SUPPORTED_LOSSES: Dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "mean_squared_error": mean_squared_error,
    "log_loss": log_loss,
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
