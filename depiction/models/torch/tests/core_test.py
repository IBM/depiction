"""Test TorchModel."""
import unittest

import numpy as np
import torchvision.models as models

from depiction.core import DataType, Task
from depiction.models.torch.core import TorchModel


class TorchModelTestCase(unittest.TestCase):
    """Test TorchModel."""

    def test_prediction(self):
        model = TorchModel(
            model=models.mobilenet_v2(pretrained=True),
            task=Task.CLASSIFICATION,
            data_type=DataType.IMAGE
        )
        image = np.random.randn(1, 3, 224, 224)
        self.assertEqual(model.predict(image).shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
