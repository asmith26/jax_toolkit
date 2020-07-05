import unittest
from unittest.mock import MagicMock

from jax import numpy as jnp

from jax_toolkit.loss import LossNotCurrentlySupportedException, get_loss_function, log_loss, mean_squared_error


class TestLogLoss(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = log_loss(jnp.array([1]), jnp.array([1]))
        self.assertEqual(0, actual_loss)
        actual_loss = log_loss(jnp.array([0]), jnp.array([0]))
        self.assertEqual(0, actual_loss)
        actual_loss = log_loss(jnp.array([1, 0]), jnp.array([1, 0]))
        self.assertEqual(0, actual_loss)
        actual_loss = log_loss(jnp.array([[0, 1], [1, 0]]), jnp.array([[0, 1], [1, 0]]))
        self.assertEqual(0, actual_loss)

        actual_loss = log_loss(jnp.array([1, 0]), jnp.array([1, 1]))
        self.assertEqual(17.269388, actual_loss)

        # Based on scikit-learnn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_classification.py#L2135
        actual_loss = log_loss(jnp.array([0, 0, 0, 1, 1, 1]), jnp.array([0.5, 0.9, 0.99, 0.1, 0.25, 0.999]))
        self.assertEqual(1.8817972, actual_loss)
        # multiclass case
        actual_loss = log_loss(
            jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])
        )
        self.assertEqual(0.69049114, actual_loss)
        # raise error if attempt to use not one hot encoded multiclass
        with self.assertRaises(ValueError) as _:
            log_loss(jnp.array([1, 0, 2]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]))
        # raise error if number of classes are not equal.
        with self.assertRaises(TypeError) as _:
            log_loss(jnp.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))


class TestMeanSquaredError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        # Below based on scikit-learn examples
        actual_loss = mean_squared_error(jnp.array([3, -0.5, 2, 7]), jnp.array([2.5, 0.0, 2, 8]))
        self.assertEqual(0.375, actual_loss)
        actual_loss = mean_squared_error(jnp.array([[0.5, 1], [-1, 1], [7, -6]]), jnp.array([[0, 2], [-1, 2], [8, -5]]))
        self.assertEqual(0.7083334, actual_loss)


class TestGetLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_loss_function(MagicMock(), loss)
