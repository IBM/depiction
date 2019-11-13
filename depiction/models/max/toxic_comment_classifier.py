"""MAX Toxic Comment Classifier."""
import numpy as np

from ...core import Task, DataType
from ..uri.rest_api.max_model import MAXModel


class ToxicCommentClassifier(MAXModel):
    """MAX Toxic Comment Classifier."""

    def __init__(self, uri):
        """
        Initialize MAX Toxic Comment Classifier.

        Args:
            uri (str): URI to access the model.
        """
        super().__init__(
            uri=uri, task=Task.MULTICLASS, data_type=DataType.TEXT
        )
        self.labels = sorted(
            self._request(method='get',
                          endpoint=self.labels_endpoint)['labels'].keys()
        )

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
                [a_prediction[label] for label in self.labels]
                for a_prediction in prediction['predictions']
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
        texts = [sample] if isinstance(sample,
                                       str) else [text for text in sample]
        return self._request(method='post', json={'text': texts})
