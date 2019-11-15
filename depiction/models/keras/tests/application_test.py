"""Test KerasApplicationModel."""
import os
import unittest

import numpy as np
from tensorflow import keras

from depiction.core import DataType, Task
from depiction.models.keras.application import KerasApplicationModel


def user_preprocessing(img_path, preprocess_input, target_size):
    """Mimic sample preparation from Keras application documentation."""
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


class KerasApplicationTestCase(unittest.TestCase):
    """Test Keras Applications."""

    def setUp(self):
        self.img_path = keras.utils.get_file(
            'elephant.jpg',
            'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Zoorashia_elephant.jpg/120px-Zoorashia_elephant.jpg'  # noqa
        )

    def test_predict_shape(self):
        """Test passing no preprocessing function."""
        model = KerasApplicationModel(
            model=keras.applications.MobileNetV2(),
            task=Task.CLASSIFICATION,
            data_type=DataType.IMAGE
        )
        image = np.random.randn(1, 224, 224, 3)
        self.assertEqual(model.predict(image).shape, (1, 1000))

    def test_predict_with_preprocessing(self):
        """Test passing of preprocessing function and arguments."""
        # default keras workflow
        img = keras.preprocessing.image.load_img(
            self.img_path, target_size=(224, 224)
        )
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        sample = keras.applications.mobilenet_v2.preprocess_input(x)
        keras_output = keras.applications.MobileNetV2().predict(sample)

        # wrapped application
        application_model = KerasApplicationModel(
            model=keras.applications.MobileNetV2(),
            task=Task.CLASSIFICATION,
            data_type=DataType.IMAGE,
            preprocessing_function=user_preprocessing,
            # kwargs passed to preprocessing_function
            preprocess_input=keras.applications.mobilenet_v2.preprocess_input,
            target_size=(224, 224)
        )
        depiction_output = application_model.predict(self.img_path)

        self.assertIs(np.array_equal(keras_output, depiction_output), True)

    def tearDown(self):
        os.remove(self.img_path)


if __name__ == "__main__":
    unittest.main()
