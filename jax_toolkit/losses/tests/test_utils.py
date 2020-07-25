import unittest
from unittest.mock import MagicMock

from jax_toolkit.losses.utils import LossNotCurrentlySupportedException, get_haiku_loss_function


class TestGetHaikuLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_haiku_loss_function(MagicMock(), loss)
