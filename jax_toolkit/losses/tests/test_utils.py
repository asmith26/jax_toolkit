import unittest
from unittest.mock import MagicMock

from jax_toolkit.losses.utils import LossNotCurrentlySupportedException, get_haiku_loss_function


class TestGetHaikuLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_haiku_loss_function(MagicMock(), loss)

    def test_supported_loss_returns_correctly(self):
        import haiku as hk
        import jax
        import jax.numpy as jnp

        def net_function(x: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([jax.nn.relu])
            predictions: jnp.ndarray = net(x)
            return predictions

        net_transform = hk.transform(net_function)
        net_transform = hk.without_apply_rng(net_transform)  # as deterministic outcome
        actual_loss_function_wrapper = get_haiku_loss_function(net_transform, loss="mean_squared_error")

        # Check works
        rng = jax.random.PRNGKey(42)
        params = net_transform.init(rng, jnp.array(0))

        self.assertEqual(0, actual_loss_function_wrapper(params, jnp.array(0), jnp.array(0)))
        self.assertEqual(0, actual_loss_function_wrapper(params, jnp.array(1), jnp.array(1)))
        self.assertEqual(1, actual_loss_function_wrapper(params, jnp.array(0), jnp.array(1)))
