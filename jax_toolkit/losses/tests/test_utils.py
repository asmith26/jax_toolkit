import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from jax_toolkit.losses.classification import log_loss
from jax_toolkit.losses.utils import LossNotCurrentlySupportedException, get_haiku_loss_function


class TestGetHaikuLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_haiku_loss_function(MagicMock(), loss)

    def test_supported_loss_returns_correctly_no_loss_kwargs(self):
        import haiku as hk

        def net_function(x: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([])
            return net(x)

        net_transform = hk.transform(net_function)
        actual_loss_function_wrapper = get_haiku_loss_function(net_transform, loss="mean_squared_error")

        # Check works
        rng = jax.random.PRNGKey(42)
        params = net_transform.init(rng, jnp.array(0))

        self.assertEqual(0, actual_loss_function_wrapper(params, jnp.array(0), jnp.array(0)))
        self.assertEqual(0, actual_loss_function_wrapper(params, jnp.array(1), jnp.array(1)))
        self.assertEqual(1, actual_loss_function_wrapper(params, jnp.array(0), jnp.array(1)))

    def test_supported_loss_returns_correctly_with_loss_kwargs(self):
        import haiku as hk

        def net_function(x: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([])
            return net(x)

        net_transform = hk.transform(net_function)
        actual_loss_function_wrapper = get_haiku_loss_function(
            net_transform, loss="sigmoid_focal_crossentropy", alpha=None, gamma=None
        )

        # Check works
        rng = jax.random.PRNGKey(42)
        params = net_transform.init(rng, jnp.array(0))

        self.assertEqual(0, actual_loss_function_wrapper(params, x=jnp.array([0]), y_true=jnp.array([0])))
        self.assertEqual(0, actual_loss_function_wrapper(params, x=jnp.array([1]), y_true=jnp.array([1])))
        # When alpha and gamma are None, it should be equal to log_loss
        self.assertEqual(
            log_loss(y_true=jnp.array([0.27]), y_pred=jnp.array([0])),
            actual_loss_function_wrapper(params, x=jnp.array([0]), y_true=jnp.array([0.27])),
        )
        self.assertEqual(
            log_loss(y_true=jnp.array([0.97]), y_pred=jnp.array([1])),
            actual_loss_function_wrapper(params, x=jnp.array([1]), y_true=jnp.array([0.97])),
        )
