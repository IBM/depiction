"""Test MAX model."""
import os
import unittest
from random import choice

from depiction.core import Task, DataType
from depiction.models.uri.rest_api.max_model import MAXModel


class ConcreteTestModel(MAXModel):

    def __init__(self, uri, task_type, data_type):
        super(ConcreteTestModel, self).__init__(uri, task_type, data_type)

    def _process_prediction(self, prediction):
        return prediction

    def _predict(self, sample, *args, **kwargs):
        return sample


class MAXModelTestCase(unittest.TestCase):
    """Test MAX model."""

    def test_initialization(self):
        model = ConcreteTestModel(
            uri='http://{}:5000'.format(
                os.environ.get('TEST_MAX_BASE', 'localhost')
            ),
            task_type=choice(list(Task)),
            data_type=choice(list(DataType))
        )
        self.assertTrue(isinstance(model.metadata, dict))
        self.assertEqual(model.metadata_endpoint, 'model/metadata')
        self.assertEqual(model.labels_endpoint, 'model/labels')
        self.assertEqual(model.endpoint, 'model/predict')


if __name__ == "__main__":
    unittest.main()
