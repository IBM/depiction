import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from depiction.core import DataType, Task
from depiction.interpreters.alibi.contrastive.cem import CEM
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
        X_expl = np.expand_dims(self.X[idx], axis=0)

        # test explainer initialization
        shape = (1, 4)  # seems first entry (batch_size) must be 1
        feature_range = (
            self.X.min(axis=0).reshape(shape) - .1,
            self.X.max(axis=0).reshape(shape) + .1
        )

        def test_mode(mode):
            interpreter = CEM(
                self.depiction_model,
                mode,
                shape,
                feature_range=feature_range,
                max_iterations=10,
                no_info_val=-1.
            )
            explanation = interpreter.interpret(X_expl, verbose=False)

            cem = interpreter.explainer
            self.assertIs(cem.model, False)
            if cem.best_attack:
                self.assertGreaterEqual(
                    set(explanation.keys()),
                    {
                        'X', 'X_pred', mode, f'{mode}_pred', 'grads_graph',
                        'grads_num'
                    }  # noqa
                )
                self.assertGreater(
                    (explanation['X'] != explanation[mode]).astype(int).sum(),
                    0
                )
                self.assertNotEqual(
                    explanation['X_pred'], explanation[f'{mode}_pred']
                )
                self.assertEqual(
                    explanation['grads_graph'].shape,
                    explanation['grads_num'].shape
                )
            else:
                self.assertGreaterEqual(
                    set(explanation.keys()), {'X', 'X_pred'}
                )

        for mode in ('PN', 'PP'):
            test_mode(mode)


if __name__ == "__main__":
    unittest.main()
