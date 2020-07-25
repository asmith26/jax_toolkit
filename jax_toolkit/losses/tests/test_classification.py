import unittest

import jax.numpy as jnp

from jax_toolkit.losses.classification import log_loss, squared_hinge


class TestLogLoss(unittest.TestCase):
    def test_binary_returns_correctly(self):
        actual_loss = log_loss(jnp.array([1]), jnp.array([1]))
        self.assertEqual(0, actual_loss)
        actual_loss = log_loss(jnp.array([0]), jnp.array([0]))
        self.assertEqual(0, actual_loss)
        actual_loss = log_loss(jnp.array([1, 0]), jnp.array([1, 0]))
        self.assertEqual(0, actual_loss)
        actual_loss = log_loss(jnp.array([1, 0]), jnp.array([1, 1]))
        self.assertEqual(17.269388, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_classification.py#L2135
        actual_loss = log_loss(jnp.array([0, 0, 0, 1, 1, 1]), jnp.array([0.5, 0.9, 0.99, 0.1, 0.25, 0.999]))
        self.assertEqual(1.8817972, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            log_loss(jnp.array([0, 1]), jnp.array([0, 1, 0]))

    def test_multiclass_returns_correctly(self):
        actual_loss = log_loss(jnp.array([[0, 1], [1, 0]]), jnp.array([[0, 1], [1, 0]]))
        self.assertEqual(0, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_classification.py#L2135
        actual_loss = log_loss(
            jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])
        )
        self.assertEqual(0.69049114, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multiclass(self):
        with self.assertRaises(TypeError) as _:
            log_loss(jnp.array([[0, 1], [1, 0]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multiclass_classes_not_equal(self):
        with self.assertRaises(TypeError) as _:
            log_loss(jnp.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_attempt_to_use_not_one_hot_encoded_multiclass(self):
        with self.assertRaises(ValueError) as _:
            log_loss(jnp.array([1, 0, 2]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]))


class TestSquaredHinge(unittest.TestCase):
    def test_binary_returns_correctly(self):
        actual_loss = squared_hinge(jnp.array([1]), jnp.array([1]))
        self.assertEqual(0, actual_loss)
        actual_loss = squared_hinge(jnp.array([-1]), jnp.array([-1]))
        self.assertEqual(0, actual_loss)
        actual_loss = squared_hinge(jnp.array([1, -1]), jnp.array([1, -1]))
        self.assertEqual(0, actual_loss)
        # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
        # /fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/metrics/tests/test_classification.py#L2037
        actual_loss = squared_hinge(jnp.array([-1, 1, 1, -1]), jnp.array([-8.5, 0.5, 1.5, -0.3]))
        self.assertEqual(0.185, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            log_loss(jnp.array([-1, 1]), jnp.array([-1, 1, 1]))

    def test_multiclass_returns_correctly(self):
        actual_loss = squared_hinge(jnp.array([[-1, 1], [1, -1]]), jnp.array([[-1, 1], [1, -1]]))
        self.assertEqual(0, actual_loss)
        # Based on tensorflow: https://github.com/tensorflow/tensorflow/blob/af7fd02ca40f362c4ac96dd064d6a2224b65d784
        # /tensorflow/python/keras/losses_test.py#L1114
        actual_loss = squared_hinge(
            jnp.array([[-1, 1, -1, 1], [-1, -1, 1, 1]]), jnp.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        )
        self.assertEqual(0.36406252, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multiclass(self):
        with self.assertRaises(TypeError) as _:
            squared_hinge(jnp.array([[-1, 1], [1, -1]]), jnp.array([[-0.2, 0.7], [0.6, -0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multiclass_classes_not_equal(self):
        with self.assertRaises(TypeError) as _:
            squared_hinge(
                jnp.array([[-1, -1, 1], [-1, -1, 1], [-1, -1, 1]]), jnp.array([[-0.2, 0.7], [0.6, -0.5], [0.4, 0.1]])
            )
