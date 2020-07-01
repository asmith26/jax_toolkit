from typing import Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp


@jax.jit
def categorical_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(y_true * -jnp.log(y_pred))


@jax.jit
def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.average((y_true - y_pred) ** 2)  # weights=sample_weight)


# =====
# UTILS
# =====
SUPPORTED_LOSSES: Dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "categorical_crossentropy": categorical_crossentropy,
    "mean_squared_error": mean_squared_error,
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
