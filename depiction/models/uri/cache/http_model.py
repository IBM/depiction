"""Abstract interface for HTTP models."""
from .cache_model import CacheModel
from ...base.utils import get_model_file


class HTTPModel(CacheModel):
    """Abstract implementation of a model cached from HTTP."""

    def __init__(self, uri, task, data_type, cache_dir, filename=None):
        """
        Initialize a HTTPModel.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
            cache_dir (str): cache directory.
            filename (str): name of the model file when cached.
                Defaults to None, a.k.a. inferring the name from
                uri.
        """
        super().__init__(
            uri=uri,
            task=task,
            data_type=data_type,
            cache_dir=cache_dir,
            filename=filename
        )

    def _get_model_file(self, filename, cache_dir):
        """
        Cache model file.

        Args:
            filename (str): name of the file.
            cache_dir (str): cache directory.

        Returns:
            str: path to the model file.
        """
        return get_model_file(filename, self.uri, cache_dir)
