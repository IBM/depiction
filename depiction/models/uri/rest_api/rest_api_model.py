"""Abstract interface for REST API models."""
from ..uri_model import URIModel


class RESTAPIModel(URIModel):
    """Abstract implementation of a REST API model."""

    def __init__(self, endpoint, uri, task, data_type):
        """
        Initalize a REST API model.

        Args:
            endpoint (str): endpoint for prediction.
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(uri=uri, task=task, data_type=data_type)
        self.endpoint = endpoint
