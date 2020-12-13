from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
@partial(jax.vmap, in_axes=(0, 0))
def _samplewise_log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Based on: https://github.com/scikit-learn/scikit-learn/blob/ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn
    /metrics/_classification.py#L2123"""
    if y_true.ndim == 0:  # If no dimension binary classification problem
        y_true = y_true.reshape(1)[:, jnp.newaxis]
        y_pred = y_pred.reshape(1)[:, jnp.newaxis]
    if y_true.shape[0] == 1:  # Reshuffle data to compute log loss correctly
        y_true = jnp.append(1 - y_true, y_true)
        y_pred = jnp.append(1 - y_pred, y_pred)

    # Clipping
    eps = 1e-15
    y_pred = y_pred.astype(jnp.float32).clip(eps, 1 - eps)

    loss = (y_true * -jnp.log(y_pred)).sum()
    return loss


@partial(jax.jit, static_argnums=(2,))
def _log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    losses = _samplewise_log_loss(y_true, y_pred)
    if normalize:
        return jnp.average(losses)  # mean loss over all samples
    return losses  # loss per sample


def log_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
    return _log_loss(y_true, y_pred, normalize)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0))
def _squared_hinge(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Based on: https://github.com/tensorflow/tensorflow/blob/af7fd02ca40f362c4ac96dd064d6a2224b65d784/tensorflow
    /python/keras/losses.py#L1324"""
    return jnp.clip(1 - y_true * y_pred, 0, None) ** 2


def squared_hinge(y_true: jnp.ndarray, y_pred: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
    losses = _squared_hinge(y_true, y_pred)
    if normalize:
        return jnp.average(losses)
    return losses


@partial(jax.jit, static_argnums=(3, 4))
@partial(jax.vmap, in_axes=(0, 0, 0, None, None))
def _sigmoid_focal_crossentropy(
    ce: jnp.ndarray, y_true: jnp.ndarray, y_pred: jnp.ndarray, alpha: float, gamma: float
) -> jnp.ndarray:
    """ Based on: https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/focal_loss.py#L90 """
    # if ce.ndim == 1:
    #     ce = ce[:, jnp.newaxis]

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = jnp.array([1.0])
    modulating_factor = jnp.array([1.0])

    if alpha:
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        modulating_factor = (1.0 - p_t) ** gamma

    losses = (alpha_factor * modulating_factor * ce).sum()
    return losses


def sigmoid_focal_crossentropy(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, alpha: float = 0.25, gamma: float = 2.0, normalize: bool = True
) -> jnp.ndarray:
    if y_true.ndim == 1:
        y_true = y_true[:, jnp.newaxis]
        y_pred = y_pred[:, jnp.newaxis]

    ce = _samplewise_log_loss(y_true, y_pred)
    losses = _sigmoid_focal_crossentropy(ce, y_true, y_pred, alpha, gamma)
    if normalize:
        return jnp.average(losses)  # mean loss over all samples
    return losses  # loss per sample
