import jax
import jax.numpy as jnp


@jax.jit
def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/_regression.py#L513 """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    denominator = ((y_true - jnp.average(y_true, axis=0)) ** 2).sum(axis=0)
    # if denominator.sum() == 0:
    #     # if numerator.sum() == 0:  # i.e. all numerator is 0, so perfect fit
    #     return jnp.array(1)
        # else:
        #     return jnp.array(0)  # i.e. constant y
    r2_scores = 1 - (numerator / denominator)[jnp.nonzero(denominator)]
    # handle nan resulting from division by 0
    if y_true.ndim == 1:  # i.e. multi-output
        return r2_scores
    else:
        return jnp.average(r2_scores[~jnp.isnan(r2_scores)])
