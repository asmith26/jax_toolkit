import unittest
from unittest.mock import MagicMock

import jax.numpy as jnp

from jax_toolkit.loss import LossNotCurrentlySupportedException, get_loss_function, log_loss, mean_squared_error, \
    max_absolute_error, median_absolute_error, r2_score, mean_absolute_error


class TestLogLoss(unittest.TestCase):
    def test_correctly_returns_and_raises_errors(self):
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
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_classification.py#L2135
        actual_loss = log_loss(jnp.array([0, 0, 0, 1, 1, 1]), jnp.array([0.5, 0.9, 0.99, 0.1, 0.25, 0.999]))
        self.assertEqual(1.8817972, actual_loss)
        # multi-class case
        actual_loss = log_loss(
            jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])
        )
        self.assertEqual(0.69049114, actual_loss)
        # raise error if attempt to use not one hot encoded multi-class
        with self.assertRaises(ValueError) as _:
            log_loss(jnp.array([1, 0, 2]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]))
        # raise error if number of classes are not equal.
        with self.assertRaises(TypeError) as _:
            log_loss(jnp.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))


class TestMeanAbsoluteError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = mean_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1, actual_loss)
        # multi-output
        actual_loss = mean_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_absolute_error(jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]),
                                            jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]]))
        self.assertEqual(0.8333334, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)


class TestMedianAbsoluteError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = median_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = median_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1, actual_loss)
        # multi-output
        actual_loss = median_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = median_absolute_error(jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]),
                                            jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]]))
        self.assertEqual(0.5, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = median_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)


class TestMaxAbsoluteError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = max_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = max_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(2, actual_loss)
        # multi-output
        actual_loss = max_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = max_absolute_error(jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]),
                                         jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]]))
        self.assertEqual(2, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = max_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)


class TestMeanSquaredError(unittest.TestCase):
    def test_returns_correctly(self):
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        # self.assertEqual(1.6666667, actual_loss)
        # multi-output
        actual_loss = mean_squared_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_error(jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]),
                                         jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]]))
        self.assertEqual(1.8333334, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_squared_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)


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


class TestGetLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_loss_function(MagicMock(), loss)
