from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

from jax_toolkit.losses.bounding_box import giou_loss
from jax_toolkit.losses.classification import log_loss, sigmoid_focal_crossentropy, squared_hinge
from jax_toolkit.losses.probabilistic import kullback_leibler_divergence
from jax_toolkit.losses.regression import (
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
)

try:
    import haiku as hk
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(
        f"{e}\n\njax_toolkit losses_utils requirements are not installed.\n\n"
        "Install with:    pip install jax_toolkit[losses_utils]"
    )


SUPPORTED_LOSSES: Dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    # Classification
    "giou_loss": giou_loss,
    # Classification
    "log_loss": log_loss,
    "squared_hinge": squared_hinge,
    "sigmoid_focal_crossentropy": sigmoid_focal_crossentropy,
    # Probabilistic
    "kullback_leibler_divergence": kullback_leibler_divergence,
    # Regression
    "mean_absolute_error": mean_absolute_error,
    "median_absolute_error": median_absolute_error,
    "max_absolute_error": max_absolute_error,
    "mean_squared_error": mean_squared_error,
    "mean_squared_log_error": mean_squared_log_error,
}


def get_haiku_loss_function(
    net_transform: hk.Transformed, loss: str, **loss_kwargs: Dict[str, float]
) -> Callable[[hk.Params, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool], jnp.ndarray]:
    try:
        loss_function = SUPPORTED_LOSSES[loss]

        @jax.jit
        def loss_function_wrapper(
            params: hk.Params, x: jnp.ndarray, y_true: jnp.ndarray, rng: jnp.ndarray = None
        ) -> jnp.ndarray:
            # rng argument can be used if net_transform.apply() is non-deterministic, and you require a "random seed"
            y_pred: jnp.ndarray = net_transform.apply(params, rng, x)
            loss_value: jnp.ndarray = loss_function(y_true, y_pred, **loss_kwargs)  # type: ignore
            return loss_value

        return loss_function_wrapper  # type: ignore
    except KeyError:
        raise LossNotCurrentlySupportedException(loss)


class LossNotCurrentlySupportedException(Exception):
    def __init__(self, loss: str):
        super().__init__(
            f"Loss={loss} is not currently supported. Currently supported losses are: {list(SUPPORTED_LOSSES.keys())}"
        )
