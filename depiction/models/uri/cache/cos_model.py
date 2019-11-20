"""Abstract interface from Cloud Object Storage (COS) models."""
import os
import copy
from minio import Minio

from .cache_model import CacheModel


class COSModel(CacheModel):
    """
    Abstract implementation of a model cached from Cloud Object Storage (COS).
    """

    def __init__(
        self, uri, task, data_type, cache_dir, filename=None, **kwargs
    ):
        """
        Initialize a COSModel.

        Args:
            uri (str): URI to access the model.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
            cache_dir (str): cache directory.
            filename (str): name of the model file when cached.
                Defaults to None, a.k.a. inferring the name from
                uri.
            kwargs (dict): key-value arguments for the Minio client.
        """
        # NOTE: check the validity of the uri provided.
        self.remote_coordinates = COSModel.parse_cos_uri(uri)
        self.minio_kwargs = copy.deepcopy(kwargs)
        # NOTE: we make sure there are no duplicated parameters
        _ = self.minio_kwargs.pop('access_key')
        _ = self.minio_kwargs.pop('secret_key')
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
        client = Minio(
            (
                f'{self.remote_coordinates["host"]}:'
                f'{self.remote_coordinates["port"]}'
            ),
            access_key=self.remote_coordinates['access_key'],
            secret_key=self.remote_coordinates['secret_key'],
            **self.minio_kwargs
        )
        filepath = os.path.join(cache_dir, self.models_subdir, filename)
        _ = client.fget_object(
            self.remote_coordinates['bucket'],
            self.remote_coordinates['filepath'],
            filepath,
            request_headers=self.minio_kwargs.get('request_headers', None)
        )
        return filepath

    @staticmethod
    def parse_cos_uri(uri):
        """
        Parse COS remote connection.

        Args:
            uri (str): cos uri.

        Returns:
            dict: a remote connection dictionary.
        """
        if not uri.startswith('s3://'):
            raise RuntimeError('Invalid S3 URI: {}'.format(uri))
        tokenized = uri[5:].split('/')
        authorization = tokenized[0]
        bucket = tokenized[1]
        filepath = '/'.join(tokenized[2:])
        keys, host = authorization.split('@')
        access_key, secret_key = keys.split(':')
        splitted_host = host.split(':')
        if len(splitted_host) > 1:
            host, port = splitted_host
        else:
            host, port = splitted_host[0], None
        return {
            'secret_key': secret_key,
            'access_key': access_key,
            'host': host,
            'port': int(port) if port else port,
            'bucket': bucket,
            'filepath': filepath
        }
