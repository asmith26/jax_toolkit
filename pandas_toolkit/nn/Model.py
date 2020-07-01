from __future__ import annotations

import copy
from typing import Callable, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix
from jax.experimental.optix import InitUpdate, OptState

from pandas_toolkit.nn.loss import get_loss_function


class Model(object):
    def __init__(
        self,
        net_function: Callable[[jnp.ndarray], jnp.ndarray],
        loss: str,
        optimizer: InitUpdate,
        example_x: jnp.ndarray,
    ):
        self.net_transform = hk.transform(net_function)
        self.optimizer = optimizer

        self.loss_function = get_loss_function(self.net_transform, loss)

        self._example_x = example_x
        self.reset_params()
        self.params: hk.Params
        self.opt_state: OptState = optimizer.init(self.params)

        self._x_columns: List[str]
        self._y_columns: List[str]

        @jax.jit
        def jitted_update(
            params: hk.Params, opt_state: OptState, x: jnp.ndarray, y: jnp.ndarray
        ) -> Tuple[hk.Params, OptState]:
            grads = jax.grad(self.loss_function)(params, x, y)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optix.apply_updates(params, updates)
            return params, opt_state

        self.jitted_update = jitted_update

        @jax.jit
        def jitted_evaluate(params: hk.Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return self.loss_function(params, x, y)

        self.jitted_evaluate = jitted_evaluate

        @jax.jit
        def jitted_predict(params: hk.Params, x: jnp.ndarray) -> jnp.ndarray:
            return self.net_transform.apply(params, x)

        self.jitted_predict = jitted_predict

    def copy(self) -> Model:
        return copy.deepcopy(self)

    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return self.jitted_evaluate(self.params, x, y)

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.jitted_predict(self.params, x)

    def _update(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        """Learning rule (e.g. stochastic gradient descent)."""
        self.params, self.opt_state = self.jitted_update(self.params, self.opt_state, x, y)

    def reset_params(self) -> None:
        self.num_epochs = 0

        rng = jax.random.PRNGKey(42)
        self.params = self.net_transform.init(rng, self._example_x)
