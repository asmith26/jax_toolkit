import jax
import jax.numpy as jnp


@jax.jit
def kullback_leibler_divergence(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on: https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/losses.py#L1598-L1636 """
    # Expand if single dimension
    if y_true.ndim == 1:
        y_true = y_true[:, jnp.newaxis]
        y_pred = y_pred[:, jnp.newaxis]

    eps = 1e-15
    y_true = jnp.clip(y_true, eps, 1)
    y_pred = jnp.clip(y_pred, eps, 1)
    loss = (y_true * jnp.log(y_true / y_pred)).sum(axis=1)
    mean_loss_all_samples = jnp.average(loss)
    return mean_loss_all_samples
