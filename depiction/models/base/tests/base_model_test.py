import unittest
from unittest import mock
from random import choice

from ..base_model import BaseModel
from ....core import Task, DataType


class ConcreteTestModel(BaseModel):
    def __init__(self, task_type, data_type):
        super(ConcreteTestModel, self).__init__(task_type, data_type)

    def predict(self, sample,*, test_kwarg):
        return sample


class BaseModelTestCase(unittest.TestCase):
    def testModelConstruction(self):
        # expected inputs
        for task_type in Task:
            for data_type in DataType:
                concrete_model = ConcreteTestModel(task_type, data_type)
                self.assertEqual(concrete_model.task, task_type)
                self.assertEqual(concrete_model.data_type, data_type)

        # unexpected inputs
        with self.assertRaises(TypeError):
            ConcreteTestModel("asad", 5)

    def testCallback(self):
        concrete_model = ConcreteTestModel(choice(list(Task)), choice(list(DataType)))

        with mock.patch.object(concrete_model, 'predict') as mock_predict:
            test_kwarg = {'test_kwarg': 'test'}
            callback = concrete_model.callback(**test_kwarg)
            test_sample = 10
            res = callback(test_sample)
            mock_predict.assert_called_with(test_sample, **test_kwarg)

    def testPredictMany(self):
        concrete_model = ConcreteTestModel(choice(list(Task)), choice(list(DataType)))

        with mock.patch.object(concrete_model, 'predict') as mock_predict:
            test_kwarg = {'test_kwarg': 'test'}
            test_samples = range(10)
            for res in concrete_model.predict_many(test_samples, **test_kwarg):
                pass
            for s in test_samples:
                mock_predict.assert_any_call(s, **test_kwarg)
