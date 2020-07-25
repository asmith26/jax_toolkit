import unittest

import jax.numpy as jnp

from jax_toolkit.metrics.regression import r2_score


class TestR2Score(unittest.TestCase):
    def test_returns_correctly(self):
        actual_score = r2_score(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(1, actual_score)
        actual_loss = r2_score(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(-1.5, actual_loss)
        # constant y_true
        actual_loss = r2_score(jnp.array([0, 0, 0]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        # multi-output
        actual_score = r2_score(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(1, actual_score)
        actual_loss = r2_score(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(0.009157509, actual_loss)
        # constant y_true
        actual_loss = r2_score(jnp.array([[0, 0, 0], [0, 0, 0]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_score = r2_score(y_true, y_pred)
        self.assertEqual(0.9951981, actual_score)
        # raise error if number of y_true and y_pred inputs don't match.
        with self.assertRaises(TypeError) as _:
            r2_score(jnp.array([0, 0]), jnp.array([0, 0, 0]))
