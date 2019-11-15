import unittest
from random import choice
from unittest import mock

from depiction.core import DataType, Task
from depiction.interpreters.base.base_interpreter import (
    AnteHocInterpreter, BaseInterpreter
)
from depiction.models.base.base_model import BaseModel


class ConcreteBaseInterpreter(BaseInterpreter):

    def interpret(self):
        return


class ConcreteAnteHocInterpreter(AnteHocInterpreter):
    SUPPORTED_TASK = set(Task)
    SUPPORTED_DATATYPE = set(DataType)

    def predict(self, sample):
        return sample

    def _fit_antehoc(self, X, y):
        return X, y

    def _fit_posthoc(self, X, y):
        return X, y

    def interpret(self, sample):
        return sample


class DummyModel(BaseModel):

    def predict(self, sample):
        return sample


class BaseInterpreterTestCase(unittest.TestCase):

    def testConstructor(self):
        with self.assertRaises(TypeError):
            interpreter = ConcreteBaseInterpreter(0)

        with self.assertRaises(ValueError):
            interpreter = ConcreteBaseInterpreter(
                DummyModel(choice(list(Task)), choice(list(DataType)))
            )


class AnteHocInterpreterTestCase(unittest.TestCase):

    def testConstructor(self):
        # - antehoc mode
        for task_type in Task:
            for data_type in DataType:
                # -- expected inputs
                interpreter = ConcreteAnteHocInterpreter(
                    AnteHocInterpreter.UsageMode.ANTE_HOC,
                    task_type=task_type,
                    data_type=data_type
                )

        # -- missing task or data
        with self.assertRaises(ValueError):
            interpreter = ConcreteAnteHocInterpreter(
                AnteHocInterpreter.UsageMode.ANTE_HOC
            )

        # - posthoc mode
        # -- calling base interpreter constructor
        def dummy_init(self, model):
            return None

        with mock.patch(
            'depiction.interpreters.base.base_interpreter.BaseModel.__init__',
            side_effect=dummy_init
        ) as mock_par_constructor:
            try:
                interpreter = ConcreteAnteHocInterpreter(
                    AnteHocInterpreter.UsageMode.POST_HOC,
                    model=DummyModel(
                        choice(list(Task)), choice(list(DataType))
                    )
                )
            except:
                pass

            mock_par_constructor.assert_called_once()

        # -- expected inputs
        for task_type in Task:
            for data_type in DataType:
                model = DummyModel(task_type, data_type)
                interpreter = ConcreteAnteHocInterpreter(
                    AnteHocInterpreter.UsageMode.POST_HOC, model=model
                )
                self.assertTrue(hasattr(interpreter, '_to_interpret'))
                self.assertEqual(interpreter._to_interpret.task, task_type)
                self.assertEqual(
                    interpreter._to_interpret.data_type, data_type
                )

        # -- missing model
        with self.assertRaises(ValueError):
            interpreter = ConcreteAnteHocInterpreter(
                AnteHocInterpreter.UsageMode.POST_HOC
            )

    def testFit(self):
        # antehoc mode
        interpreter = ConcreteAnteHocInterpreter(
            AnteHocInterpreter.UsageMode.ANTE_HOC,
            task_type=choice(list(Task)),
            data_type=choice(list(DataType))
        )

        with mock.patch.object(interpreter, '_fit_antehoc') as mock_fit:
            interpreter.fit(0, 0)
            mock_fit.assert_called_with(0, 0)

        model = DummyModel(choice(list(Task)), choice(list(DataType)))
        interpreter = ConcreteAnteHocInterpreter(
            AnteHocInterpreter.UsageMode.POST_HOC, model=model
        )

        with mock.patch.object(interpreter, '_fit_posthoc') as mock_fit:
            interpreter.fit(0, 0)
            mock_fit.assert_called_with(0, 0)


if __name__ == "__main__":
    unittest.main()
