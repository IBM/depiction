"""Counterfactual explanation test."""
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from depiction.core import DataType, Task
from depiction.interpreters.alibi.counterfactual import Counterfactual
from depiction.models.base.base_model import BaseModel


class SKLearnModel(BaseModel):

    def __init__(self, clf):
        super().__init__(Task.CLASSIFICATION, DataType.TABULAR)
        self.clf = clf

    def predict(self, X):
        return self.clf.predict_proba(X)


class CEMTestCase(unittest.TestCase):
    """Matching test for implementation source.
    
    Reference:
        https://github.com/SeldonIO/alibi/blob/92e8048ea2f4e4ef57b6874fa854b90de8ed9602/alibi/explainers/tests/test_cem.py#L9  # noqa
    """

    def setUp(self):
        dataset = load_iris()

        # scale dataset
        dataset.data = (dataset.data -
                        dataset.data.mean(axis=0)) / dataset.data.std(axis=0)

        # define train and test set
        self.X, self.Y = dataset.data, dataset.target

        # fit random forest to training data
        np.random.seed(0)
        clf = LogisticRegression(solver='liblinear')
        clf.fit(self.X, self.Y)

        # define Model
        self.depiction_model = SKLearnModel(clf)

    def testInterpretation(self):
        """Matching test to source implementation."""

        # instance to be explained
        idx = 0
        X_to_interpret = np.expand_dims(self.X[idx], axis=0)

        # test explainer initialization
        shape = (1, 4)  # seems first entry (batch_size) must be 1

        interpreter = Counterfactual(self.depiction_model, shape)
        explanation = interpreter.interpret(X_to_interpret)

        counterfactual = interpreter.explainer
        self.assertEqual(
            counterfactual.return_dict['meta']['name'],
            counterfactual.__class__.__name__
        )
        self.assertEqual(explanation['cf']['X'].shape, X_to_interpret.shape)
        self.assertEqual(len(explanation['all']), counterfactual.max_lam_steps)


if __name__ == "__main__":
    unittest.main()
