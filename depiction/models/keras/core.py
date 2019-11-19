"""Core module for Keras models."""
import copy

from ..base.base_model import BaseModel


class KerasModel(BaseModel):
    """Keras model wrapper."""

    def __init__(self, model, task, data_type):
        """
        Initialize a Model.

        Args:
            model (torch.nn.Module): model to wrap.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(task=task, data_type=data_type)
        self._model = model
        self._predict_kwargs = {
            'batch_size': None,
            'verbose': 0,
            'steps': None,
            'callbacks': None
        }

    def _prepare_sample(self, sample):
        """
        Prepare sample for the model.

        Args:
            sample (np.ndarray): an input sample for the model.

        Returns:
            np.ndarray: a numpy array representing the prepared sample.
        """
        return sample

    def predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (np.ndarray): an input sample for the model.
            args (list): list of arguments for prediction.
            kwargs (dict): list of key-value arguments for prediction.

        Returns:
            np.ndarray: a prediction for the model on the given sample.
        """
        predict_kwargs = copy.deepcopy(self._predict_kwargs)
        predict_kwargs.update(**kwargs)
        return self._model.predict(
            self._prepare_sample(sample), *args, **predict_kwargs
        )
