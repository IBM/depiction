"""Abstract interface for MAX models."""
import os
from abc import abstractmethod

from .rest_api_model import RESTAPIModel


class MAXModel(RESTAPIModel):
    """
    Abstract implementation of a MAX model.

    For a complete model list see here:
    https://developer.ibm.com/exchanges/models/all/.
    """

    def __init__(self, uri, task, data_type):
        """
        Initalize a MAX model.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        self.base_endpoint = 'model'
        super().__init__(
            endpoint=os.path.join(self.base_endpoint, 'predict'),
            uri=uri,
            task=task,
            data_type=data_type
        )
        self.metadata_endpoint = os.path.join(self.base_endpoint, 'metadata')
        self.labels_endpoint = os.path.join(self.base_endpoint, 'labels')
        self.metadata = self._request(
            method='get', endpoint=self.metadata_endpoint
        )

    @abstractmethod
    def _process_prediction(self, prediction):
        """
        Process json prediction response.

        Args:
            prediction (dict): json prediction response.

        Returns:
            np.array: numpy array representing the prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def _predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (object): an input sample for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            a prediction for the model on the given sample.
        """
        raise NotImplementedError

    def predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (object): an input sample for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            a prediction for the model on the given sample.
        """
        return self._process_prediction(self._predict(sample, *args, **kwargs))
