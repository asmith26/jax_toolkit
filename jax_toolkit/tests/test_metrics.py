import unittest

import jax.numpy as jnp

from jax_toolkit.metrics import r2_score


class TestR2Score(unittest.TestCase):
    def test_returns_correctly(self):
        actual_score = r2_score(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(1, actual_score)
        # multi-output
        actual_score = r2_score(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(1, actual_score)
        # # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1

        # from sklearn.metrics import r2_score
        # import numpy as np
        # r2_score(np.arange(50), np.arange(50)+1)
        actual_score = r2_score(y_true, y_pred)
        self.assertEqual(0.9951981, actual_score)
