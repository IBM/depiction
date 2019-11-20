"""Abstract interface for MAX models."""
import os

from .rest_api_model import RESTAPIModel


class MAXModel(RESTAPIModel):
    """
    Abstract implementation of a MAX model.

    For a complete model list see here:
    https://developer.ibm.com/exchanges/models/all/.
    """

    def __init__(self, uri, task, data_type):
        """
        Initialize a MAX model.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        self.base_endpoint = 'model'
        super().__init__(
            endpoint=os.path.join(self.base_endpoint, 'predict'),
            uri=uri,
            task=task,
            data_type=data_type
        )
        self.metadata_endpoint = os.path.join(self.base_endpoint, 'metadata')
        self.labels_endpoint = os.path.join(self.base_endpoint, 'labels')
        self.metadata = self._request(
            method='get', endpoint=self.metadata_endpoint
        )
