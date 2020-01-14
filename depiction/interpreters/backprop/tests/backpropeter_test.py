import warnings
warnings.simplefilter(action='ignore')

import torch
import inspect
import unittest
import numpy as np
from torch import nn
from tensorflow.keras.layers import Dense
from unittest.mock import patch
import torch.nn.functional as F
from tensorflow.keras.models import Sequential

from depiction.core import DataType, Task
from depiction.models.torch.core import TorchModel
from depiction.models.keras.core import KerasModel
from depiction.models.base.base_model import BaseModel
from depiction.interpreters.backprop.backpropeter import BackPropeter


INPUT_SZ = 5
OUTPUT_SZ = 7


class DummyModel(BaseModel):
    def predict(self, sample):
        return None


class DummyTorchModel(nn.Module):
    """
    From https://github.com/pytorch/captum#getting-started
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(INPUT_SZ, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, OUTPUT_SZ)

        # initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, -4.0 + 3.0*np.float(INPUT_SZ)).view(3, INPUT_SZ))
        self.lin1.bias = nn.Parameter(torch.zeros(1,3))
        self.lin2.weight = nn.Parameter(torch.arange(-4.0, -4.0 + 3.0*np.float(OUTPUT_SZ)).view(OUTPUT_SZ, 3))
        self.lin2.bias = nn.Parameter(torch.ones(1,OUTPUT_SZ))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))


class BackPropeterTestCase(unittest.TestCase):
    """
    Test class for back-propagation like attribution methods
    """
    def setUp(self):
        self._available_model_types = list(BackPropeter.METHODS.keys())
        self._test_data_type = np.random.choice([d for d in DataType])
        self._test_task_type = Task.CLASSIFICATION

        torch_method_name = np.random.choice(list(BackPropeter.METHODS['torch'].keys()))
        self._torch = {
            'model': TorchModel(DummyTorchModel(), self._test_task_type, self._test_data_type),
            'method_name': torch_method_name,
            'method_class': BackPropeter.METHODS['torch'][torch_method_name],
            'method_classname': BackPropeter.METHODS['torch'][torch_method_name].__name__
        }

        self._keras = {
            'model': KerasModel(self._build_keras_model(), self._test_task_type, self._test_data_type),
            'method_name': np.random.choice(list(BackPropeter.METHODS['keras'].keys()))
        }

    def testMethodCheck(self):
        model_type = np.random.choice(self._available_model_types)
        with self.assertRaises(ValueError):
            BackPropeter._check_supported_method(model_type, 'dummy_test_algo')

    def _build_keras_model(self):        
        """
        From https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        """
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=INPUT_SZ, activation='relu'))
        model.add(Dense(OUTPUT_SZ, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def testConstructor(self):
        # test invalid model
        with self.assertRaises(ValueError):
            BackPropeter(DummyModel(self._test_task_type, self._test_data_type), 'doesnt matter')

        # test torch model
        with patch.object(BackPropeter, '_check_supported_method') as mock_check:
            class_constructor = 'depiction.interpreters.backprop.backpropeter.{}.__init__'.format(
                self._torch['method_classname'])
            with patch(class_constructor, return_value=None) as mock_init:
                interpreter = BackPropeter(self._torch['model'], self._torch['method_name'])
                self.assertIs(type(interpreter._explainer), self._torch['method_class'])
                mock_check.assert_called_once_with('torch', self._torch['method_name'])
                mock_init.assert_called_once_with(self._torch['model']._model)

        # test keras model
        with patch.object(BackPropeter, '_check_supported_method') as mock_check:
            interpreter = BackPropeter(self._keras['model'], self._keras['method_name'])
            mock_check.assert_called_once_with('keras', self._keras['method_name'])
            
    def testInterpret(self):
        batch_size = np.random.choice(10) + 1

        # test torch model
        for m in BackPropeter.METHODS['torch'].keys():
            interpreter = BackPropeter(self._torch['model'], self._torch['method_name'])
            args = inspect.signature(interpreter._explainer.attribute).parameters.keys()

            x = np.random.rand(batch_size, INPUT_SZ)
            output = torch.tensor(self._torch['model'].predict(x))
            output = F.softmax(output, dim=1)
            prediction_score, pred_label_idx = torch.topk(output, 1, dim=1)
            pred_label_idx = pred_label_idx.squeeze()

            # -- -- without delta
            interpret_kwargs = {
                'baselines': torch.zeros(batch_size, INPUT_SZ),
                'target': pred_label_idx,
            } 
            allowed_kwargs = interpret_kwargs.keys() & set(args)
            res = interpreter.interpret(x, explanation_configs={arg: interpret_kwargs[arg] for arg in allowed_kwargs})
            self.assertIsInstance(res, torch.Tensor)

            # -- -- with delta
            if interpreter._explainer.has_convergence_delta():
                interpret_kwargs['return_convergence_delta'] = True
                allowed_kwargs = interpret_kwargs.keys() & set(args)
                res = interpreter.interpret(x, explanation_configs={arg: interpret_kwargs[arg] for arg in allowed_kwargs})
                self.assertIsInstance(res, tuple)
                self.assertIsInstance(res[0], torch.Tensor)            
                self.assertIsInstance(res[1], torch.Tensor)

        # test keras model
        for m in BackPropeter.METHODS['keras'].keys():
            interpreter = BackPropeter(self._keras['model'], m)
            x = np.random.rand(batch_size, INPUT_SZ)
            res1 = interpreter.interpret([x])
            res2 = interpreter.interpret(x)