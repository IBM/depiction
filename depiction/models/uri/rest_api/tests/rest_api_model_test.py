"""Test REST API model."""
import unittest
from random import choice

from .....core import Task, DataType
from ..rest_api_model import RESTAPIModel


class ConcreteTestModel(RESTAPIModel):

    def __init__(self, endpoint, uri, task_type, data_type):
        super(ConcreteTestModel,
              self).__init__(endpoint, uri, task_type, data_type)

    def predict(self, sample, *, test_kwarg):
        return sample


class RESTAPIModelTestCase(unittest.TestCase):
    """Test REST API model."""

    def test_initialization(self):
        model = ConcreteTestModel(
            endpoint='predict',
            uri='http://localhost:5000',
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
