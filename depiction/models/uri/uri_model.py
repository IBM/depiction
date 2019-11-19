"""Abstract interface for URI models."""
from ..base.base_model import BaseModel


class URIModel(BaseModel):
    """Abstract implementation of a URI model."""

    def __init__(self, uri, task, data_type):
        """
        Initialize a URIModel.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(task=task, data_type=data_type)
        self.uri = uri
