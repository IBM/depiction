"""Abstract interface for REST API models."""
import os
import requests
from abc import abstractmethod

from ..uri_model import URIModel


class RESTAPIModel(URIModel):
    """Abstract implementation of a REST API model."""

    def __init__(self, endpoint, uri, task, data_type):
        """
        Initialize a REST API model.

        Args:
            endpoint (str): endpoint for prediction.
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(uri=uri, task=task, data_type=data_type)
        self.endpoint = endpoint

    def _request(self, method, endpoint=None, **kwargs):
        """
        Perform a request to self.uri.

        Args:
            method (str): request method.
            endpoint (str): request endpoint.
                Defaults to None, a.k.a. use self.endpoint.
            kwargs (dict): key-value arguments for requests.request.

        Returns:
            dict: response dictionary.
        """
        response = requests.request(
            method=method,
            url=os.path.join(
                self.uri, endpoint if endpoint else self.endpoint
            ),
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    @abstractmethod
    def _process_prediction(self, prediction):
        """
        Process json prediction response.

        Args:
            prediction (dict): json prediction response.

        Returns:
            np.ndarray: numpy array representing the prediction.
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