"""Abstract interface for MAX models."""
from .rest_api_model import RESTAPIModel


class MAXModel(RESTAPIModel):
    """
    Abstract implementation of a MAX model.

    For a complete model list see here:
    https://developer.ibm.com/exchanges/models/all/.
    """

    def __init__(self, uri, task, data_type):
        """
        Initalize a MAX model.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(
            endpoint='predict', uri=uri, task=task, data_type=data_type
        )
