import unittest

import jax.numpy as jnp

from jax_toolkit.losses.regression import (
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
)


class TestMeanAbsoluteError(unittest.TestCase):
    def test_single_output_returns_correctly(self):
        actual_loss = mean_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            mean_absolute_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))

    def test_multioutput_returns_correctly(self):
        actual_loss = mean_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_absolute_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(0.8333334, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multioutput(self):
        with self.assertRaises(TypeError) as _:
            mean_absolute_error(jnp.array([[0, 1], [1, 2]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multioutput_outputs_not_equal(self):
        with self.assertRaises(TypeError) as _:
            mean_absolute_error(jnp.array([[1, 2, 3], [10, 11, 12]]), jnp.array([[0.2, 0.7], [0.6, 0.5]]))


class TestMedianAbsoluteError(unittest.TestCase):
    def test_single_output_returns_correctly(self):
        actual_loss = median_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = median_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = median_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            median_absolute_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))

    def test_multioutput_returns_correctly(self):
        actual_loss = median_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = median_absolute_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(0.5, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multioutput(self):
        with self.assertRaises(TypeError) as _:
            median_absolute_error(jnp.array([[0, 1], [1, 2]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multioutput_outputs_not_equal(self):
        with self.assertRaises(TypeError) as _:
            median_absolute_error(jnp.array([[1, 2, 3], [10, 11, 12]]), jnp.array([[0.2, 0.7], [0.6, 0.5]]))


class TestMaxAbsoluteError(unittest.TestCase):
    def test_single_output_returns_correctly(self):
        actual_loss = max_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = max_absolute_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(2, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = max_absolute_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            max_absolute_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))

    def test_multioutput_returns_correctly(self):
        actual_loss = max_absolute_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = max_absolute_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(2, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multioutput(self):
        with self.assertRaises(TypeError) as _:
            max_absolute_error(jnp.array([[0, 1], [1, 2]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multioutput_outputs_not_equal(self):
        with self.assertRaises(TypeError) as _:
            max_absolute_error(jnp.array([[1, 2, 3], [10, 11, 12]]), jnp.array([[0.2, 0.7], [0.6, 0.5]]))


class TestMeanSquaredError(unittest.TestCase):
    def test_single_output_returns_correctly(self):
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(1.6666667, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_squared_error(y_true, y_pred)
        self.assertEqual(1, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            mean_squared_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))

    def test_multioutput_returns_correctly(self):
        actual_loss = mean_squared_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(1.8333334, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multioutput(self):
        with self.assertRaises(TypeError) as _:
            mean_squared_error(jnp.array([[0, 1], [1, 2]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multioutput_outputs_not_equal(self):
        with self.assertRaises(TypeError) as _:
            mean_squared_error(jnp.array([[1, 2, 3], [10, 11, 12]]), jnp.array([[0.2, 0.7], [0.6, 0.5]]))


class TestMeanSquaredLogError(unittest.TestCase):
    def test_single_output_returns_correctly(self):
        actual_loss = mean_squared_log_error(jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_log_error(jnp.array([0, 1, 2]), jnp.array([0, 0, 0]))
        self.assertEqual(0.56246734, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_regression.py#L25
        y_true = jnp.arange(50)
        y_pred = y_true + 1
        actual_loss = mean_squared_log_error(y_true, y_pred)
        self.assertEqual(0.019151635, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            mean_squared_log_error(jnp.array([0, 0]), jnp.array([0, 0, 0]))

    def test_multioutput_returns_correctly(self):
        actual_loss = mean_squared_log_error(jnp.array([[0, 1, 2], [0, 1, 2]]), jnp.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(0, actual_loss)
        actual_loss = mean_squared_log_error(
            jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 0, 0]]), jnp.array([[1, 2, 0, 1], [4, 3, 1, 1], [0, 0, 0, 1]])
        )
        self.assertEqual(0.36393017, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multioutput(self):
        with self.assertRaises(TypeError) as _:
            mean_squared_log_error(jnp.array([[0, 1], [1, 2]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multioutput_outputs_not_equal(self):
        with self.assertRaises(TypeError) as _:
            mean_squared_log_error(jnp.array([[1, 2, 3], [10, 11, 12]]), jnp.array([[0.2, 0.7], [0.6, 0.5]]))
