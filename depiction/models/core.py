"""Core utilities and classes for model handling."""
from pathlib import Path

from tensorflow.keras.utils import get_file


def get_model_file(filename, origin, cache_dir):
    """
    Downloads a file from a URL if it not already in the cache.
    """
    return get_file(filename,
                    origin,
                    cache_subdir='models',
                    cache_dir=cache_dir)


class Model(object):
    """Abstract implementation of a model."""

    def __init__(self, task, filename, origin,
                 cache_dir=Path(__file__).parents[1] / 'cache'):
        """
        Initalize a Model.

        Arguments:
            task (Task): task type.
            filename (str): filename of the model file.
            origin (str): url of the file to download.
            cache_dir (str or Path): directory to save the file, default
                is 'cache' under module root. Without write access files are
                stored under '/tmp/.keras'
        """

        self.task = task
        self.model_path = get_model_file(filename, origin, cache_dir)

    def callback(self, sample, **kwargs):
        """
        Return a callback function that can be called directly on the samples.
        The additional arguments are wrapped and embedded in the function call.

        Arguments:
            sample (object): an input sample for the model.
            kwargs (dict): list of key-value arguments.

        Returns:
            a function taking a sample an input and returning the prediction.
        """
        return lambda sample: self.predict(sample, **kwargs)

    def predict(self, sample, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        arameters.

        Arguments:
            sample (object): an input sample for the model.
            kwargs (dict): list of key-value arguments.

        Returns
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


class TextModel(object):
    """Abstract implementation of a text model."""

    def __init__(self, task, filename, origin, *args, **kwargs):
        """
        Initalize a TextModel.

        Arguments:
            task (Task): task type.
            filename (str): filename of the model file.
            origin (str): url of the file to download.
        """
        super(TextModel).__init__(
            self, task, filename, origin, *args, **kwargs
        )

    def get_language(self):
        """
        Get a spacy.language.Language object for the model.

        Returns:
            a spacy.language.Language.
        """
        raise NotImplementedError
