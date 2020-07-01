import unittest
from unittest.mock import MagicMock

import jax.numpy as jnp

from pandas_toolkit.nn.loss import LossNotCurrentlySupportedException, get_loss_function, mean_squared_error


class TestGetLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_loss_function(MagicMock(), loss)


class TestMeanSquaredError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        # Below based on scikit-learn examples
        actual_loss = mean_squared_error(jnp.array([3, -0.5, 2, 7]), jnp.array([2.5, 0.0, 2, 8]))
        self.assertEqual(0.375, actual_loss)
        actual_loss = mean_squared_error(jnp.array([[0.5, 1], [-1, 1], [7, -6]]), jnp.array([[0, 2], [-1, 2], [8, -5]]))
        self.assertEqual(0.7083333, actual_loss)
