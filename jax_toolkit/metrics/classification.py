import jax
import jax.numpy as jnp


@jax.jit
def _intersection_over_union(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py#L4 """
    intersection = jnp.sum(jnp.abs(y_true * y_pred))
    sum_ = jnp.sum(jnp.abs(y_true) + jnp.abs(y_pred))
    jac = (intersection) / (sum_ - intersection)
    return jac


def intersection_over_union(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    if (y_true < 0).any() or (y_true > 1).any() or (y_pred < 0).any() or (y_pred > 1).any():
        raise ValueError("Currently support only when y_true, y_pred in range [0, 1]")
    if (y_true == y_pred).all():
        return 1

    return _intersection_over_union(y_true, y_pred)
