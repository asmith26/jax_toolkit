import jax
import jax.numpy as jnp


@jax.jit
def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = jnp.average(jnp.abs(y_pred - y_true), axis=0)
    mean_loss_all_samples = jnp.average(loss)
    return mean_loss_all_samples


@jax.jit
def median_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = jnp.median(jnp.abs(y_pred - y_true), axis=0)
    mean_loss_all_samples = jnp.average(loss)
    return mean_loss_all_samples


@jax.jit
def max_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    loss = jnp.abs(y_pred - y_true).max(axis=0)
    mean_loss_all_samples = jnp.average(loss)
    return mean_loss_all_samples


@jax.jit
def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.average((y_true - y_pred) ** 2)
