"""Abstract interface for file system models."""
from .cache_model import CacheModel


class FileSystemModel(CacheModel):
    """Abstract implementation of a model stored on file system."""

    def __init__(self, uri, task, data_type):
        """
        Initialize a FileSystemModel.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(
            uri=uri,
            task=task,
            data_type=data_type,
            cache_dir=None,
            filename=None
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
        return self.uri
