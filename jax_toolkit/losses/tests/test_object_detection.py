import unittest

import jax.numpy as jnp

from jax_toolkit.losses.object_detection import giou_loss


class TestGiouLoss(unittest.TestCase):
    def test_returns_correctly(self):
        boxes1 = jnp.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
        boxes2 = jnp.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        expected_loss = jnp.array([1.075, 1.9333334])
        actual_loss = giou_loss(boxes1, boxes2)
        self.assertTrue((expected_loss == actual_loss).all())

    #     actual_loss = log_loss(jnp.array([1]), jnp.array([1]))
    #     self.assertEqual(0, actual_loss)
    #     actual_loss = log_loss(jnp.array([0]), jnp.array([0]))
    #     self.assertEqual(0, actual_loss)
    #     actual_loss = log_loss(jnp.array([1, 0]), jnp.array([1, 0]))
    #     self.assertEqual(0, actual_loss)
    #     actual_loss = log_loss(jnp.array([1, 0]), jnp.array([1, 1]))
    #     self.assertEqual(17.269388, actual_loss)
    #     # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
    #     # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_classification.py#L2135
    #     actual_loss = log_loss(jnp.array([0, 0, 0, 1, 1, 1]), jnp.array([0.5, 0.9, 0.99, 0.1, 0.25, 0.999]))
    #     self.assertEqual(1.8817972, actual_loss)
    #
    # def test_raises_when_number_of_samples_not_equal(self):
    #     with self.assertRaises(TypeError) as _:
    #         log_loss(jnp.array([0, 1]), jnp.array([0, 1, 0]))
    #
    # def test_multiclass_returns_correctly(self):
    #     actual_loss = log_loss(jnp.array([[0, 1], [1, 0]]), jnp.array([[0, 1], [1, 0]]))
    #     self.assertEqual(0, actual_loss)
    #     # Based on scikit-learn: https://github.com/scikit-learn/scikit-learn/blob
    #     # /ffbb1b4a0bbb58fdca34a30856c6f7faace87c67/sklearn/metrics/tests/test_classification.py#L2135
    #     actual_loss = log_loss(
    #         jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])
    #     )
    #     self.assertEqual(0.69049114, actual_loss)
    #
    # def test_raises_when_number_of_samples_not_equal_multiclass(self):
    #     with self.assertRaises(TypeError) as _:
    #         log_loss(jnp.array([[0, 1], [1, 0]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))
    #
    # def test_raises_when_number_of_multiclass_classes_not_equal(self):
    #     with self.assertRaises(TypeError) as _:
    #         log_loss(jnp.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), jnp.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]))
    #
    # def test_raises_when_attempt_to_use_not_one_hot_encoded_multiclass(self):
    #     with self.assertRaises(ValueError) as _:
    #         log_loss(jnp.array([1, 0, 2]), jnp.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]))
