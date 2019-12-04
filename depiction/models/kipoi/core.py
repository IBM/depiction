"""Core module for Keras models."""
import copy

import kipoi
from ..base.base_model import BaseModel


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


class KipoiModel(BaseModel):
    """To use Kipoi models via its Python API.

    See https://github.com/kipoi/kipoi/blob/master/notebooks/python-api.ipynb.

    Take care that Kipoi models might define additional dependencies."""

    def __init__(
        self, model, task, data_type, source='kipoi', with_dataloader=False,
        preprocessing_function=identity,
        postprocessing_function=identity,
        preprocessing_kwargs={},
        postprocessing_kwargs={}
    ):
        """
        Initialize a KipoiModel via `kipoi.get_model`.

        Args:
            model (string): kipoi model name.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
            source (str):  kipoi model source name. Defaults to 'kipoi'.
            with_dataloader (bool): if True, the kipoi models' default
                dataloader is loaded to `model.default_dataloader` and the
                pipeline at `model.pipeline` enabled. Defaults to False.
            preprocessing_function (callable): function to preprocess samples.
            **preprocessing_kwargs (dict): keyword arguments passed to
            preprocessing function.
            postprocessing_function (callable): function to postprocess output
                of kipois `predict_on_batch`.
            **postprocessing_kwargs (dict): keyword arguments passed to
                postprocessing function.

        The processing functions default to the identity function.
        """
        super().__init__(task=task, data_type=data_type)
        self.preprocessing_function = preprocessing_function
        self.preprocessing_kwargs = copy.deepcopy(preprocessing_kwargs)
        self.postprocessing_function = postprocessing_function
        self.postprocessing_kwargs = copy.deepcopy(postprocessing_kwargs)
        self.model = kipoi.get_model(
            model, source=source, with_dataloader=with_dataloader
        )

    def _prepare_sample(self, sample):
        """
        Prepare sample for the model.

        Args:
            sample (np.ndarray): an input sample for the model.

        Returns:
            output of preprocessing function representing the sample.
        """
        return self.preprocessing_function(
            sample, **self.preprocessing_kwargs
        )

    def predict(self, sample):
        """
        Run the model for inference on a given sample. The sample is
        preprocessed and output postprocessed.

        Args:
            sample (np.ndarray): an input sample for the model.

        Returns:
            np.ndarray: a prediction for the model on the given sample.
        """
        return self.postprocessing_function(
            self.model.predict_on_batch(self._prepare_sample(sample)),
            **self.postprocessing_kwargs
        )
