import unittest
from unittest.mock import MagicMock

import haiku as hk
import jax.numpy as jnp
import pandas as pd
from jax.nn import relu

import pandas_toolkit.nn
from pandas_toolkit.nn import _get_batch, _get_num_batches
from pandas_toolkit.nn.Model import Model
from pandas_toolkit.utils.custom_types import Batch


class TestGetNumBatches(unittest.TestCase):
    def test_returns_1_when_batch_size_is_None(self):
        batch_size = None
        num_rows = MagicMock()
        actual_batch_number = _get_num_batches(num_rows, batch_size)
        expected_batch_number = 1
        self.assertEqual(expected_batch_number, actual_batch_number)

    def test_returns_correctly_when_batch_size_perfectly_divides_num_rows(self):
        batch_size = 2
        num_rows = 10
        actual_batch_number = _get_num_batches(num_rows, batch_size)
        expected_batch_number = 5
        self.assertEqual(expected_batch_number, actual_batch_number)

    def test_returns_correctly_when_batch_size_does_not_perfectly_divide_num_rows(self):
        batch_size = 3
        num_rows = 10
        actual_batch_number = _get_num_batches(num_rows, batch_size)
        expected_batch_number = 4
        self.assertEqual(expected_batch_number, actual_batch_number)


class TestGetBatch(unittest.TestCase):
    def test_returns_correctly(self):
        df_train = pd.DataFrame({"x": [0, 1, 2], "y": [10, 11, 12]})
        batch_size = 2

        batch_number = 0
        actual_batch = _get_batch(df_train, batch_number, batch_size, x_columns=["x"], y_columns=["y"])
        expected_batch = Batch(x=jnp.array([[0], [1]]), y=jnp.array([[10], [11]]))
        self.assertTrue(
            (actual_batch.x == expected_batch.x).all(),
            f"expected_batch.x={expected_batch.x}!={actual_batch.x}=actual_batch.x",
        )
        self.assertTrue(
            (actual_batch.y == expected_batch.y).all(),
            f"expected_batch.y={expected_batch.y}!={actual_batch.y}=actual_batch.y",
        )

        batch_number = 1
        actual_batch = _get_batch(df_train, batch_number, batch_size, x_columns=["x"], y_columns=["y"])
        expected_batch = Batch(x=jnp.array([[2]]), y=jnp.array([[12]]))
        self.assertTrue(
            (actual_batch.x == expected_batch.x).all(),
            f"expected_batch.x={expected_batch.x}!={actual_batch.x}=actual_batch.x",
        )
        self.assertTrue(
            (actual_batch.y == expected_batch.y).all(),
            f"expected_batch.y={expected_batch.y}!={actual_batch.y}=actual_batch.y",
        )


class TestWorkflow(unittest.TestCase):
    def test_simple_relu_net(self):
        # Train/Validation data
        df_train = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

        def net_function(x: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([relu])
            predictions: jnp.ndarray = net(x)
            return predictions

        df_train = df_train.nn.init(
            x_columns=["x"], y_columns=["y"], net_function=net_function, loss="mean_squared_error"
        )

        for _ in range(10):  # num_epochs
            # df = df.nn.augment()
            # df = df.nn.shuffle()
            df_train = df_train.nn.update()

        self.assertTrue(isinstance(df_train.model, Model), "df.model is not of type pandas_toolkit.nn.Model")

        # Test data
        df_test = pd.DataFrame({"x": [-1, 0, 1], "y": [0, 0, 1]})
        df_test.model = df_train.nn.get_model()
        actual_loss = df_test.nn.evaluate()
        expected_loss = 0
        self.assertEqual(expected_loss, actual_loss)

        # New data
        df_new = pd.DataFrame({"x": [-10, -5, 22]})
        df_new.model = df_train.nn.get_model()  # ToDo probably be pandas_toolkit.nn.load_model(model_path)

        actual_predictions = df_new.nn.predict()
        expected_predictions = jnp.array([0, 0, 22])
        self.assertTrue(
            (actual_predictions == expected_predictions).all(),
            f"expected_predictions={expected_predictions}!={actual_predictions}=actual_predictions",
        )
