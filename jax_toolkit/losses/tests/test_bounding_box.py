import unittest

import jax.numpy as jnp

from jax_toolkit.losses.bounding_box import giou_loss


class TestGiouLoss(unittest.TestCase):
    def test_exact_match_returns_0(self):
        boxes1 = jnp.array([[3.0, 4.0, 7.0, 8.0]])
        boxes2 = jnp.array([[3.0, 4.0, 7.0, 8.0]])
        expected_loss = jnp.array([0])
        actual_loss = giou_loss(boxes1, boxes2)
        self.assertEqual(expected_loss, actual_loss)

    def test_1_box_returns_correctly(self):
        # Based on: https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/tests/giou_loss_test
        # .py#L64
        boxes1 = jnp.array([[4.0, 3.0, 7.0, 5.0]])
        boxes2 = jnp.array([[3.0, 4.0, 6.0, 8.0]])
        expected_loss = jnp.array(1.075)
        actual_loss = giou_loss(boxes1, boxes2)
        self.assertEqual(expected_loss, actual_loss)

    def test_2_boxes_returns_correctly(self):
        # Based on: https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/tests/giou_loss_test
        # .py#L42
        boxes1 = jnp.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
        boxes2 = jnp.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        expected_loss = jnp.array(1.5041666)
        actual_loss = giou_loss(boxes1, boxes2)
        self.assertEqual(expected_loss, actual_loss)

    def test_raises_when_number_of_samples_not_equal(self):
        with self.assertRaises(ValueError) as _:
            boxes1 = jnp.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
            boxes2 = jnp.array([[4.0, 3.0, 7.0, 5.0]])
            _ = giou_loss(boxes1, boxes2)
