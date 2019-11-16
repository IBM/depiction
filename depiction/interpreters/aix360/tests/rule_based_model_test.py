import unittest
import numpy as np
from random import choice
from unittest import mock

from aix360.algorithms.rbm import (
    BooleanRuleCG, BRCGExplainer, GLRMExplainer, LinearRuleRegression,
    LogisticRuleRegression
)
from pandas import DataFrame

from depiction.core import DataType, Task
from depiction.interpreters.aix360.rule_based_model import RuleAIX360
from depiction.models.base.base_model import BaseModel


class DummyModel(BaseModel):

    def predict(self, sample):
        return np.array([choice([0, 1]) for _ in range(sample.shape[0])])


class RuleAIX360TestCase(unittest.TestCase):

    def setUp(self):
        self.X = np.random.randn(100, 10)
        self.y = (np.random.randn(100) > 0.).astype(int)

    def _build_posthoc_interpreter(self):
        model = DummyModel(
            choice(list(RuleAIX360.SUPPORTED_TASK)),
            choice(list(RuleAIX360.SUPPORTED_DATATYPE))
        )
        interpreter = RuleAIX360(
            choice(list(RuleAIX360.AVAILABLE_INTERPRETERS)),
            X=self.X,
            model=model
        )
        return interpreter

    def _build_antehoc_interpreter(self):
        interpreter = RuleAIX360(
            choice(list(RuleAIX360.AVAILABLE_INTERPRETERS)), self.X, y=self.y
        )
        return interpreter

    def testConstructor(self):
        # test error for wrong model
        NOT_SUPPORTED_TASKS = [
            t for t in set(Task) for T in RuleAIX360.SUPPORTED_TASK
            if not (t <= T)
        ]
        NOT_SUPPORTED_TYPES = list(
            set(DataType).difference(RuleAIX360.SUPPORTED_DATATYPE)
        )

        wrong_model = DummyModel(
            choice(NOT_SUPPORTED_TASKS), choice(NOT_SUPPORTED_TYPES)
        )

        with self.assertRaises(ValueError):
            RuleAIX360(
                choice(list(RuleAIX360.AVAILABLE_INTERPRETERS)),
                X=self.X,
                model=wrong_model
            )

        # test error for not supported interpreter
        with self.assertRaises(ValueError):
            RuleAIX360('', X=self.X, y=self.y)

        # test error for not supported GLRM regressor
        with self.assertRaises(ValueError):
            RuleAIX360('glrm_bubu', X=self.X, y=self.y)

        # test correctly chosen glrm and regressor
        valid_glrm = [
            i for i in RuleAIX360.AVAILABLE_INTERPRETERS if 'glrm' in i
        ]
        interpreter = RuleAIX360(choice(valid_glrm), X=self.X, y=self.y)
        self.assertTrue(isinstance(interpreter.explainer, GLRMExplainer))
        self.assertTrue(
            isinstance(interpreter.regressor, LogisticRuleRegression)
            or isinstance(interpreter.regressor, LinearRuleRegression)
        )
        self.assertFalse(interpreter._fitted)

        # -- test correctness of ante-hoc model
        self.assertEqual(interpreter.usage_mode, RuleAIX360.UsageMode.ANTE_HOC)
        self.assertTrue(
            Task.check_support(interpreter.task, RuleAIX360.SUPPORTED_TASK)
        )
        self.assertTrue(interpreter.data_type in RuleAIX360.SUPPORTED_DATATYPE)

        # test brcg model
        interpreter = RuleAIX360('brcg', X=self.X, y=self.y)
        self.assertTrue(isinstance(interpreter.explainer, BRCGExplainer))
        self.assertTrue(isinstance(interpreter.regressor, BooleanRuleCG))
        self.assertFalse(interpreter._fitted)

        # test with right model
        interpreter = self._build_posthoc_interpreter()
        self.assertEqual(interpreter.usage_mode, RuleAIX360.UsageMode.POST_HOC)
        self.assertFalse(interpreter._fitted)

    def testFit(self):
        # test fit antehoc called correctly
        interpreter = self._build_antehoc_interpreter()

        with mock.patch.object(
            interpreter, '_fit_antehoc'
        ) as mock_fit_antehoc:
            interpreter.fit(0, 0)
            mock_fit_antehoc.assert_called_once()

        # test fit posthoc called correctly
        interpreter = self._build_posthoc_interpreter()

        with mock.patch.object(
            interpreter, '_fit_posthoc'
        ) as mock_fit_posthoc:
            interpreter.fit(0, 0)
            mock_fit_posthoc.assert_called_once()

    def testFitAntehoc(self):
        interpreter = self._build_antehoc_interpreter()

        with mock.patch.object(
            interpreter.explainer, 'fit'
        ) as mock_explainer_fit:
            interpreter.fit(0, 0)
            mock_explainer_fit.assert_called_once()

    def testFitPosthoc(self):
        interpreter = self._build_posthoc_interpreter()

        with mock.patch.object(
            interpreter._to_interpret, 'predict'
        ) as mock_predict:
            with mock.patch.object(
                interpreter, '_fit_antehoc'
            ) as mock_fit_antehoc:
                interpreter.fit(0)

                mock_predict.assert_called_once()
                mock_fit_antehoc.assert_called_once()

        with mock.patch.object(
            interpreter._to_interpret, 'predict'
        ) as mock_predict:
            with mock.patch.object(
                interpreter, '_fit_antehoc'
            ) as mock_fit_antehoc:
                preprocess = mock.MagicMock()

                interpreter.fit(0, preprocess)
                preprocess.assert_called_once()
                preprocess.assert_called_with(0)

        with mock.patch.object(
            interpreter._to_interpret, 'predict', return_value=2
        ) as mock_predict:
            with mock.patch.object(
                interpreter, '_fit_antehoc'
            ) as mock_fit_antehoc:
                postprocess = mock.MagicMock()

                interpreter.fit(0, postprocess_y=postprocess)
                postprocess.assert_called_once()
                postprocess.assert_called_with(2)

    def testInterpret(self):
        builder = choice(
            [self._build_posthoc_interpreter, self._build_antehoc_interpreter]
        )
        interpreter = builder()

        with mock.patch.object(
            interpreter.explainer, 'explain'
        ) as mock_explain:
            with mock.patch.object(
                interpreter, '_visualize_explanation'
            ) as mock_visualize:
                e = interpreter.interpret()

                mock_explain.assert_called_once()
                mock_visualize.assert_called_once()
                self.assertTrue(e, interpreter.explanation)

        with mock.patch.object(
            interpreter.explainer, 'explain'
        ) as mock_explain:
            with mock.patch.object(
                interpreter, '_save_explanation'
            ) as mock_save:
                e = interpreter.interpret(path='')

                mock_explain.assert_called_once()
                mock_save.assert_called_once()
                self.assertTrue(e, interpreter.explanation)

    def testVisualize(self):
        """
        TODO(phineasng): think if it's possible or make sense to test this
        """
        pass

    def testSave(self):
        builder = choice(
            [self._build_posthoc_interpreter, self._build_antehoc_interpreter]
        )
        interpreter = builder()

        # test DataFrame
        df = DataFrame()
        with mock.patch.object(df, 'to_pickle') as mock_to_pickle:
            interpreter._save_explanation(df, path='')
            mock_to_pickle.assert_called_with('')

        exp = object()
        module_name = 'depiction.interpreters.aix360.rule_based_model'
        with mock.patch('{}.open'.format(module_name)) as mock_open:
            with mock.patch('{}.pickle.dump'.format(module_name)) as mock_dump:
                interpreter._save_explanation(exp, path='')
                mock_open.assert_called_once()
                mock_open.assert_called_with('', 'wb')
                mock_dump.assert_called_once()

    def testPredict(self):
        builder = choice(
            [self._build_posthoc_interpreter, self._build_antehoc_interpreter]
        )
        interpreter = builder()

        with mock.patch.object(
            interpreter.explainer, 'predict'
        ) as mock_predict:
            interpreter.predict(0)
            mock_predict.assert_called_once()
            mock_predict.assert_called_with(0)


if __name__ == "__main__":
    unittest.main()
