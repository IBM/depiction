"""Abstract interface for models."""
from pathlib import Path
from abc import ABC, abstractmethod

from ...core import Task, DataType


class BaseModel(ABC):
    """Abstract implementation of a model."""

    def __init__(self, task, data_type):
        """
        Initalize a Model.

        Arguments:
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        if not isinstance(task, Task) or not isinstance(data_type, DataType):
            raise TypeError

        self.task = task
        self.data_type = data_type

    def callback(self, **kwargs):
        """
        Return a callback function that can be called directly on the samples.
        The additional arguments are wrapped and embedded in the function call.

        Arguments:
            kwargs (dict): list of key-value arguments.

        Returns:
            a function taking a sample an input and returning the prediction.
        """
        return lambda sample: self.predict(sample, **kwargs)

    @abstractmethod
    def predict(self, sample, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        arameters.

        Arguments:
            sample (object): an input sample for the model.
            kwargs (dict): list of key-value arguments.

        Returns:
            a prediction for the model on the given sample.
        """
        raise NotImplementedError

    def predict_many(self, samples, **kwargs):
        """
        Run the model for inference on the given samples and with the provided
        parameters.

        Arguments:
            samples (Iterable): input samples for the model.
            kwargs (dict): list of key-value arguments.

        Returns:
            a generator of predictions.
        """
        for sample in samples:
            yield self.predict(sample, **kwargs)
