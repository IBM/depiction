"""Abstract interface for URI models."""
import os
from abc import abstractclassmethod

from ..uri_model import URIModel
from ...base.utils import MODELS_SUBDIR


class CacheModel(URIModel):
    """Abstract implementation of a cached URI model."""

    def __init__(self, uri, task, data_type, cache_dir, filename=None):
        """
        Initialize a CacheModel.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
            cache_dir (str): cache directory.
            filename (str): name of the model file when cached.
                Defaults to None, a.k.a. inferring the name from
                uri.
        """
        super().__init__(uri=uri, task=task, data_type=data_type)
        self.models_subdir = MODELS_SUBDIR
        if filename is None:
            filename = os.path.basename(self.uri)
        self.cache_dir = cache_dir
        self.filename = filename
        self.model_path = self._get_model_file(self.filename, self.cache_dir)

    @abstractclassmethod
    def _get_model_file(self, filename, cache_dir):
        """
        Cache model file.

        Args:
            filename (str): name of the file.
            cache_dir (str): cache directory.

        Returns:
            str: path to the model file.
        """
        raise NotImplementedError
