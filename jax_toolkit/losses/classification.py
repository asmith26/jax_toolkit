from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def _samplewise_log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Based on: https://github.com/scikit-learn/scikit-learn/blob/ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn
    /metrics/_classification.py#L2123"""
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

    # Renormalize (i.e. make probabilities)
    y_pred /= y_pred.sum(axis=1)[:, jnp.newaxis]
    loss = (y_true * -jnp.log(y_pred)).sum(axis=1)
    return loss


@jax.jit
def log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = _samplewise_log_loss(y_true, y_pred)
    mean_loss_all_samples = jnp.average(loss)
    return mean_loss_all_samples


@jax.jit
def squared_hinge(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Based on: https://github.com/tensorflow/tensorflow/blob/af7fd02ca40f362c4ac96dd064d6a2224b65d784/tensorflow
    /python/keras/losses.py#L1324"""
    return jnp.average(jnp.clip(1 - y_true * y_pred, 0, None) ** 2)


@partial(jax.jit, static_argnums=(2, 3))
def _sigmoid_focal_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray, alpha: float, gamma: float) -> jnp.ndarray:
    """ Based on: https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/focal_loss.py#L90 """
    # If single dimension, assume binary classification problem
    if y_true.ndim == 1:
        y_true = y_true[:, jnp.newaxis]
        y_pred = y_pred[:, jnp.newaxis]

    # if gamma and gamma < 0:
    #       raise ValueError("Value of gamma should be greater than or equal to zero")

    # Compute the _samplewise_log_loss
    ce = _samplewise_log_loss(y_true, y_pred)
    if ce.ndim == 1:
        ce = ce[:, jnp.newaxis]

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = jnp.array([1.0])
    modulating_factor = jnp.array([1.0])

    if alpha:
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        modulating_factor = (1.0 - p_t) ** gamma

    # compute the final loss and return
    loss = (alpha_factor * modulating_factor * ce).sum(axis=1)
    mean_loss_all_samples = jnp.average(loss)
    return mean_loss_all_samples


def sigmoid_focal_crossentropy(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, alpha: float = 0.25, gamma: float = 2.0
) -> jnp.ndarray:
    return _sigmoid_focal_crossentropy(y_true, y_pred, alpha, gamma)
