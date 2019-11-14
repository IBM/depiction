"""Test KerasModel."""
import unittest

import numpy as np
from tensorflow import keras

from depiction.core import DataType, Task
from depiction.models.keras.core import KerasModel


class KerasModelTestCase(unittest.TestCase):
    """Test KerasModel."""

    def test_prediction(self):
        model = KerasModel(
            model=keras.applications.MobileNetV2(),
            task=Task.CLASSIFICATION,
            data_type=DataType.IMAGE
        )
        image = np.random.randn(1, 224, 224, 3)
        self.assertEqual(model.predict(image).shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
