"""Binarized model."""
import numpy as np

from .base_model import BaseModel
from ...core import Task


class BinarizedClassifier(BaseModel):

    def __init__(self, model, data_type, label_index):
        """
        Initialize a Model.

        Args:
            model (torch.nn.Module): model to wrap.
            data_type (depiction.core.DataType): data type.
            label_index (int): index of the label to consider as positive.
        """
        super(BinarizedClassifier, self).__init__(Task.BINARY, data_type)
        self.model = model
        self.label_index = label_index

    def predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (np.ndarray): an input sample for the model.
            args (list): list of arguments for prediction.
            kwargs (dict): list of key-value arguments for prediction.

        Returns:
            int: 1 or 0 depending on the highest logit.
        """
        y = self.model.predict(sample, *args, **kwargs)
        return (np.argmax(y, axis=1) == self.label_index).astype(np.int)
