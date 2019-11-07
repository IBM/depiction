"""Abstract interface for MAX models."""
import itertools
import numpy as np
from ABC import abstractmethod
from multiprocessing import Pool

from .rest_api_model import RESTAPIModel


class MAXModel(RESTAPIModel):
    """
    Abstract implementation of a MAX model.

    For a complete model list see here:
    https://developer.ibm.com/exchanges/models/all/.
    """

    def __init__(self, uri, task, data_type, processes=1):
        """
        Initalize a MAX model.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
            processess (int): process used in predict_many.
                Defaults to 1.
        """
        super().__init__(
            endpoint='predict', uri=uri, task=task, data_type=data_type
        )
        self.metadata_endpoint = 'metadata'
        self.labels_endpoint = 'labels'
        self.processes = processes

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
        arameters.

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
        arameters.

        Args:
            sample (object): an input sample for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            a prediction for the model on the given sample.
        """
        return self._process_prediction(self._predict(sample, *args, **kwargs))

    def predict_many(self, samples, *args, **kwargs):
        """
        Run the model for inference on the given samples and with the provided
        parameters.

        Args:
            samples (Iterable): input samples for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            np.array: a generator of predictions.
        """
        with Pool(processes=self.processes) as pool:
            predictions = pool.starmap(
                lambda sample, args, kwargs: np.
                expand_dims(self.predict(sample, args, kwargs), axis=0),
                zip(samples, itertools.repeat(args), itertools.repeat(kwargs))
            )
        return np.vstack(predictions)
