"""Abstract interface for models."""
from pathlib import Path
from abc import ABC, abstractmethod

from ...core import Task, DataType


class BaseModel(ABC):
    """Abstract implementation of a model."""

    def __init__(self, task, data_type):
        """
        Initalize a Model.

        Args:
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        if not isinstance(task, Task) or not isinstance(data_type, DataType):
            raise TypeError("Inputs must be valid Task and DataType types!")

        self.task = task
        self.data_type = data_type

    def callback(self, *argv, **kwargs):
        """
        Return a callback function that can be called directly on the samples.
        The additional arguments are wrapped and embedded in the function call.

        Args:
            kwargs (dict): list of key-value arguments.

        Returns:
            a function taking a sample an input and returning the prediction.
        """
        return lambda sample: self.predict(sample, *argv, **kwargs)

    @abstractmethod
    def predict(self, sample, *argv, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        arameters.

        Args:
            sample (object): an input sample for the model.
            kwargs (dict): list of key-value arguments.

        Returns:
            a prediction for the model on the given sample.
        """
        raise NotImplementedError

    def predict_many(self, samples, *argv, **kwargs):
        """
        Run the model for inference on the given samples and with the provided
        parameters.

        Args:
            samples (Iterable): input samples for the model.
            kwargs (dict): list of key-value arguments.

        Returns:
            a generator of predictions.
        """
        for sample in samples:
            yield self.predict(sample, *argv, **kwargs)


class TrainableModel(BaseModel):
    """Interface for trainable models."""

    @abstractmethod
    def fit(self, *argv, **kwargs):
        raise NotImplementedError