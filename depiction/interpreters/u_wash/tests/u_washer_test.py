import unittest
import numpy as np
from unittest import mock
from random import choice
import spacy

from ....core import Task, DataType
from ....models.base.base_model import BaseModel
from ..u_washer import UWasher


class DummyModel(BaseModel):

    def predict(self, sample):
        return sample


class UWasherTestCase(unittest.TestCase):

    def testConstructor(self):

        def dummy_init(*args, **kwargs):
            return None

        def test_routine(explainer_key, explainer_cls, model):
            with mock.patch(
                'depiction.interpreters.u_wash.u_washer.{}.__init__'.
                format(explainer_cls),
                side_effect=dummy_init
            ) as mock_constructor:
                interpreter = UWasher(explainer_key, model)
                self.assertTrue(interpreter.model is model)
                mock_constructor.assert_called_once()

        task_type = Task.CLASSIFICATION

        data_type = DataType.TEXT
        model = DummyModel(task_type, data_type)
        test_routine('lime', 'LimeTextExplainer', model)
        test_routine('anchors', 'AnchorText', model)

        data_type = DataType.TABULAR
        model = DummyModel(task_type, data_type)
        test_routine('lime', 'LimeTabularExplainer', model)
        test_routine('anchors', 'AnchorTabularExplainer', model)

    def testInterpret(self):
        task_type = Task.CLASSIFICATION
        data_type = DataType.TABULAR
        model = DummyModel(task_type, data_type)

        class DummyExplanation:
            SHOW_IN_NOTEBOOK = False
            PATH = ''

            def show_in_notebook(self):
                self.SHOW_IN_NOTEBOOK = True

            def save_to_file(self, path):
                self.PATH = path

        def dummy_interpret(*args, **kwargs):
            return DummyExplanation()

        interpreter = UWasher(
            'lime', model, **{'training_data': np.array([[0, 0], [1, 1]])}
        )
        test_config = {'dummy_config': 10}
        test_callback_kwargs = {}
        dummy_sample = [10, 15]

        with mock.patch.object(
            interpreter.explainer,
            'explain_instance',
            side_effect=dummy_interpret
        ) as mock_explain:
            explanation = interpreter.interpret(
                dummy_sample, test_callback_kwargs, test_config, path=None
            )
            mock_explain.assert_called_once()
            self.assertEqual(explanation.SHOW_IN_NOTEBOOK, True)
            self.assertEqual(explanation.PATH, '')

        with mock.patch.object(
            interpreter.explainer,
            'explain_instance',
            side_effect=dummy_interpret
        ) as mock_explain:
            dummy_path = 'tests'
            explanation = interpreter.interpret(
                dummy_sample,
                test_callback_kwargs,
                test_config,
                path=dummy_path
            )
            mock_explain.assert_called_once()
            self.assertEqual(explanation.SHOW_IN_NOTEBOOK, False)
            self.assertEqual(explanation.PATH, dummy_path)