"""Core module for keras applications."""
from .core import KerasModel


def identity(sample, *args, **kwargs):
    """
    Apply identity.

    Args:
        sample (np.ndarray): an input sample for the model.

    Returns:
        np.ndarray: output of preprocessing function representing
            the sample.
    """
    return sample


class KerasApplicationModel(KerasModel):
    """Keras application wrapper."""

    def __init__(
        self,
        model,
        task,
        data_type,
        preprocessing_function=identity,
        *args,
        **kwargs
    ):
        """
        Initialize a KerasApplicationModel.

        Args:
            model (keras): model to wrap.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
            preprocessing_function (callable): function to preprocess samples.
            *args (list): arguments passed to preprocessing function.
            **kwargs (dict): keyword arguments passed to preprocessing
                function.
        """
        super().__init__(model=model, task=task, data_type=data_type)
        self.preprocessing_function = preprocessing_function
        self.preprocessing_args = args
        self.preprocessing_kwargs = kwargs

    def _prepare_sample(self, sample):
        """
        Prepare sample for the model.

        Args:
            sample (np.ndarray): an input sample for the model.

        Returns:
            output of preprocessing function representing the sample.
        """
        return self.preprocessing_function(
            sample, *self.preprocessing_args, **self.preprocessing_kwargs
        )
