import unittest

import jax.numpy as jnp

from jax_toolkit.metrics.classification import intersection_over_union


class TestIntersectionOverUnion(unittest.TestCase):
    def test_binary_returns_correctly(self):
        actual_loss = intersection_over_union(jnp.array([0]), jnp.array([0]))
        self.assertEqual(1, actual_loss)
        actual_score = intersection_over_union(jnp.array([0]), jnp.array([1]))
        self.assertEqual(0, actual_score)
        actual_loss = intersection_over_union(jnp.array([1, 0]), jnp.array([1, 0]))
        self.assertEqual(1, actual_loss)
        actual_loss = intersection_over_union(jnp.array([0, 1, 1]), jnp.array([0, 1, 0]))
        self.assertEqual(0.5, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(TypeError) as _:
            intersection_over_union(jnp.array([0, 1]), jnp.array([0, 1, 0]))

    def test_raises_when_values_below_0(self):
        with self.assertRaises(ValueError) as _:
            intersection_over_union(jnp.array([-1]), jnp.array([1]))
        with self.assertRaises(ValueError) as _:
            intersection_over_union(jnp.array([1]), jnp.array([-1]))

    def test_raises_when_values_above_1(self):
        with self.assertRaises(ValueError) as _:
            intersection_over_union(jnp.array([2]), jnp.array([0]))
        with self.assertRaises(ValueError) as _:
            intersection_over_union(jnp.array([0]), jnp.array([2]))

    def test_multiclass_returns_correctly(self):
        actual_loss = intersection_over_union(jnp.array([[0, 1], [1, 0]]), jnp.array([[0, 1], [1, 0]]))
        self.assertEqual(1, actual_loss)
        # Based on: https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7
        # /sklearn/metrics/tests/test_classification.py#L1257
        actual_score = intersection_over_union(jnp.array([[0, 1, 1], [1, 0, 1]]), jnp.array([[0, 0, 1], [1, 0, 1]]))
        self.assertEqual(0.75, actual_score)

    def test_raises_when_number_of_samples_not_equal_multiclass(self):
        with self.assertRaises(TypeError) as _:
            intersection_over_union(jnp.array([[0, 1], [1, 0]]), jnp.array([[0, 1], [1, 0], [0, 1]]))

    def test_raises_when_number_of_multiclass_classes_not_equal(self):
        with self.assertRaises(TypeError) as _:
            intersection_over_union(jnp.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), jnp.array([[0, 1], [1, 0], [0, 1]]))
