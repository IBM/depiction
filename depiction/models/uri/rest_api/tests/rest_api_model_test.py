"""Test REST API model."""
import os
import unittest
from random import choice

from depiction.core import DataType, Task
from depiction.models.uri.rest_api.rest_api_model import RESTAPIModel


class ConcreteTestModel(RESTAPIModel):

    def __init__(self, endpoint, uri, task_type, data_type):
        super(ConcreteTestModel,
              self).__init__(endpoint, uri, task_type, data_type)

    def _process_prediction(self, prediction):
        return prediction

    def _predict(self, sample, *args, **kwargs):
        return sample


class RESTAPIModelTestCase(unittest.TestCase):
    """Test REST API model."""

    def test_initialization(self):
        model = ConcreteTestModel(
            endpoint='predict',
            uri='http://{}:5000'.format(
                os.environ.get('TEST_REST_API', 'localhost')
            ),
            task_type=choice(list(Task)),
            data_type=choice(list(DataType))
        )
        self.assertTrue(
            model._request(method='get', endpoint='model/metadata'), dict
        )
        self.assertTrue(
            model._request(method='get', endpoint='model/labels'), dict
        )
        self.assertTrue(
            model._request(
                method='post',
                endpoint='model/predict',
                json={'text': ['a test.', 'another test.']}
            ), dict
        )


if __name__ == "__main__":
    unittest.main()
