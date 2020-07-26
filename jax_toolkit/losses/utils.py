from typing import Callable, Dict

import jax
import jax.numpy as jnp

from jax_toolkit.losses.classification import log_loss, squared_hinge
from jax_toolkit.losses.probabilistic import kullback_leibler_divergence
from jax_toolkit.losses.regression import (
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
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
    "log_loss": log_loss,
    "squared_hinge": squared_hinge,
    "kullback_leibler_divergence": kullback_leibler_divergence,
    "mean_absolute_error": mean_absolute_error,
    "median_absolute_error": median_absolute_error,
    "max_absolute_error": max_absolute_error,
    "mean_squared_error": mean_squared_error,
}


def get_haiku_loss_function(
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
