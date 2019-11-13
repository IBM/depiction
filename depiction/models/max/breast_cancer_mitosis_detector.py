"""MAX Breast Cancer Mitosis Detector Keras Model."""
import imageio
import numpy as np
from io import BytesIO

from ...core import Task, DataType
from ..uri.rest_api.max_model import MAXModel


class BreastCancerMitosisDetector(MAXModel):
    """MAX Breast Cancer Mitosis Detector Keras Model."""

    def __init__(self, uri):
        """
        Initialize MAX Breast Cancer Mitosis Detector Keras Model.

        Args:
            uri (str): URI to access the model.
        """
        super().__init__(uri=uri, task=Task.BINARY, data_type=DataType.IMAGE)
        self.labels = ['non mitotic', 'mitotic']

    def _process_prediction(self, prediction):
        """
        Process json prediction response.

        Args:
            prediction (dict): json prediction response.

        Returns:
            np.ndarray: numpy array representing the prediction.
        """
        return np.array(
            [
                [
                    1.0 - a_prediction['probability'],
                    a_prediction['probability']
                ] for a_prediction in prediction['predictions']
            ]
        )

    def _predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (object): an input sample for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            dict: a prediction for the model on the given sample.
        """
        # NOTE: create a buffer containing the image
        buffer = BytesIO()
        imageio.imwrite(buffer, sample, format='png')
        buffer.seek(0, 0)
        return self._request(method='post', files={'image': buffer})
