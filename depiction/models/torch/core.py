"""Core module for PyTorch models."""
import torch

from ..base.base_model import BaseModel


class TorchModel(BaseModel):
    """PyTorch model wrapper."""

    def __init__(self, model, task, data_type, double=False):
        """
        Initialize a TorchModel.

        Args:
            model (torch.nn.Module): model to wrap.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(task=task, data_type=data_type)
        self._model = model
        self._double = double

    def _prepare_sample(self, sample):
        """
        Prepare sample for the model.

        Args:
            sample (np.ndarray): an input sample for the model.

        Returns:
            torch.tensor: a tensor representing the sample.
        """
        if self._double:
            return torch.from_numpy(sample).double()
        return torch.from_numpy(sample).float()

    def predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (np.ndarray): an input sample for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            np.ndarray: a prediction for the model on the given sample.
        """
        if self._double:
            self._model = self._model.double().eval()
            return self._model(self._prepare_sample(sample).double(), **kwargs).detach().numpy()
        self._model = self._model.float().eval()
        return self._model(self._prepare_sample(sample).float(), **kwargs).detach().numpy()
