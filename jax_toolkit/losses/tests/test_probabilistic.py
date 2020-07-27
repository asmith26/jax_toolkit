import unittest

import jax.numpy as jnp

from jax_toolkit.losses.probabilistic import kullback_leibler_divergence


class TestKullbackLeiblerDivergence(unittest.TestCase):
    def test_single_output_returns_correctly(self):
        actual_loss = kullback_leibler_divergence(jnp.array([1]), jnp.array([1]))
        self.assertEqual(0, actual_loss)
        actual_loss = kullback_leibler_divergence(jnp.array([1, 2]), jnp.array([1, 2]))
        self.assertEqual(0, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            kullback_leibler_divergence(jnp.array([1, 1]), jnp.array([0, 1, 1]))

    def test_multiclass_returns_correctly(self):
        actual_loss = kullback_leibler_divergence(jnp.array([[1, 1]]), jnp.array([[1, 1]]))
        self.assertEqual(0, actual_loss)
        # Based on tensorflow: https://github.com/tensorflow/tensorflow/blob/af7fd02ca40f362c4ac96dd064d6a2224b65d784
        # /tensorflow/python/keras/losses_test.py#L1459-L1465
        y_pred = jnp.asarray([0.4, 0.9, 0.12, 0.36, 0.3, 0.4]).reshape((2, 3))
        y_true = jnp.asarray([0.5, 0.8, 0.12, 0.7, 0.43, 0.8]).reshape((2, 3))
        actual_loss = kullback_leibler_divergence(y_true, y_pred)
        self.assertEqual(0.59607387, actual_loss)

    def test_raises_when_number_of_samples_not_equal_multiclass(self):
        with self.assertRaises(TypeError) as _:
            kullback_leibler_divergence(jnp.array([[0, 1], [1, 0]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))

    def test_raises_when_number_of_multiclass_classes_not_equal(self):
        with self.assertRaises(TypeError) as _:
            kullback_leibler_divergence(
                jnp.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]])
            )
