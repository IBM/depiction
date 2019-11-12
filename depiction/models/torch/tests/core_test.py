"""Test TorchModel."""
import unittest
import numpy as np
import torchvision.models as models

from ....core import Task, DataType
from ..core import TorchModel


class TorchModelTestCase(unittest.TestCase):
    """Test TorchModel."""

    def test_prediction(self):
        model = TorchModel(
            model=models.mobilenet_v2(pretrained=True),
            task=Task.CLASSIFICATION,
            data_type=DataType.IMAGE
        )
        image = np.random.randn(1, 3, 224, 224)
        self.assertEqual(
            model.predict(image).shape,
            (1, 1000)
        )
