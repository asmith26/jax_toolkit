import unittest

import jax.numpy as jnp
from jax_toolkit.losses.regression import mean_absolute_error, median_absolute_error, max_absolute_error, \
    mean_squared_error


class TestMeanAbsoluteError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = mean_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1, actual_loss)
        # multi-output
        actual_loss = mean_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_absolute_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(0.8333334, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)
        # raise error if number of y_true and y_pred inputs don't match.
        with self.assertRaises(TypeError) as _:
            mean_absolute_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))


class TestMedianAbsoluteError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = median_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = median_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1, actual_loss)
        # multi-output
        actual_loss = median_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = median_absolute_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(0.5, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = median_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)
        # raise error if number of y_true and y_pred inputs don't match.
        with self.assertRaises(TypeError) as _:
            median_absolute_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))


class TestMaxAbsoluteError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = max_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = max_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(2, actual_loss)
        # multi-output
        actual_loss = max_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = max_absolute_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(2, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = max_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)
        # raise error if number of y_true and y_pred inputs don't match.
        with self.assertRaises(TypeError) as _:
            max_absolute_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))


class TestMeanSquaredError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1.6666667, actual_loss)
        # multi-output
        actual_loss = mean_squared_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(1.8333334, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_squared_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)
        # raise error if number of y_true and y_pred inputs don't match.
        with self.assertRaises(TypeError) as _:
            mean_squared_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))
