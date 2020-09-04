import jax.numpy as jnp
from jax.ops import index, index_update


# @jax.jit
def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
    /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/_regression.py#L513"""
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    denominator = ((y_true - jnp.average(y_true, axis=0)) ** 2).sum(axis=0)
    if denominator.sum() == 0:  # i.e. constant y_true
        if numerator.sum() == 0:
            return jnp.array(1)  # i.e. all numerator is 0, so perfect fit
        else:
            # arbitrary set to zero (following scikit-learn) to avoid -inf scores, having a
            # constant y_true is not interesting for scoring a regression anyway
            return jnp.array(0)
    r2_scores = 1 - (numerator / denominator)
    if y_true.ndim == 1:
        return r2_scores
    else:  # i.e. multi-output
        # handle nan resulting from division by 0
        r2_scores = index_update(r2_scores, index[jnp.isnan(r2_scores)], 0)
        return jnp.average(r2_scores)
