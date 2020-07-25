import jax
import jax.numpy as jnp


@jax.jit
def log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/_classification.py#L2123 """
    # If single dimension, assume binary classification problem
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
def squared_hinge(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on tensorflow: https://github.com/tensorflow/tensorflow/blob/af7fd02ca40f362c4ac96dd064d6a2224b65d784
        /tensorflow/python/keras/losses.py#L1324 """
    return jnp.average(jnp.clip(1 - y_true * y_pred, 0, None) ** 2)
