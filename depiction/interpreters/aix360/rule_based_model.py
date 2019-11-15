"""
Wrapper around the rule based models implemented in the AIX360 framework

References:
- https://github.com/IBM/AIX360
- Wei, D., Dash, S., Gao, T. & Gunluk, O.. (2019). Generalized Linear Rule Models. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:6687-6696
- Dash, S., Gunluk, O., & Wei, D. (2018). Boolean decision rules via column generation. In Advances in Neural Information Processing Systems (pp. 4655-4665).
"""
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG
from aix360.algorithms.rbm import (
    GLRMExplainer, LogisticRuleRegression, LinearRuleRegression
)
from aix360.algorithms.rbm import FeatureBinarizer

from ...core import Task, DataType
from ..base.base_interpreter import AnteHocInterpreter, ExplanationType


class RuleAIX360(AnteHocInterpreter):
    _AVAILABLE_RULE_REGRESSORS = {'logistic', 'linear'}

    SUPPORTED_TASK = {Task.BINARY}
    SUPPORTED_DATATYPE = {DataType.TABULAR}

    AVAILABLE_INTERPRETERS = {'brcg'}.union(
        {'glrm_{}'.format(i)
         for i in _AVAILABLE_RULE_REGRESSORS}
    )

    EXPLANATION_TYPE = ExplanationType.GLOBAL

    def __init__(self, explainer, X, model=None, y=None, regressor_params={}):
        """
        Constructor. For a description of the missing arguments,
        please refer to the AnteHocInterpreter.

        Args:
            - explainer (str): name of the explainer to use.
            - X (np.ndarray or pd.DataFrame): data to explain.
            - model (depiction.models.base.BaseModel): a model to interpret.
                Defaults to None, a.k.a. ante-hoc.
            - y (np.ndarray): binary labels for X.
                Defaults to None, a.k.a. post-hoc.
            - regressor_params (dict): parameters for the regressor.s
        """
        is_post_hoc = y is None
        is_ante_hoc = model is None
        if is_ante_hoc and is_post_hoc:
            raise RuntimeError(
                'Make sure you pass a model (post-hoc) or labels (ante-hoc)'
            )
        if model is None:
            super(RuleAIX360, self).__init__(
                AnteHocInterpreter.UsageMode.ANTE_HOC,
                task_type=Task.BINARY,
                data_type=DataType.TABULAR
            )
        else:
            super(RuleAIX360, self).__init__(
                AnteHocInterpreter.UsageMode.POST_HOC, model=model
            )

        if 'glrm' in explainer:
            regressor = explainer.split('_')[1]
            if regressor == 'logistic':
                self.regressor = LogisticRuleRegression(**regressor_params)
            elif regressor == 'linear':
                self.regressor = LinearRuleRegression(**regressor_params)
            else:
                raise ValueError(
                    "Regressor '{}' not supported! Available regressors: {}".
                    format(regressor, self._AVAILABLE_RULE_REGRESSORS)
                )
            self.explainer = GLRMExplainer(self.regressor)
        elif explainer == 'brcg':
            self.regressor = BooleanRuleCG(**regressor_params)
            self.explainer = BRCGExplainer(self.regressor)
        else:
            raise ValueError(
                "Interpreter '{}' not supported! Available interpreters: {}".
                format(explainer, self.AVAILABLE_INTERPRETERS)
            )

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.X = X
        self.y = y
        self.binarizer = FeatureBinarizer(negations=True)
        self.X_binarized = self.binarizer.fit_transform(self.X)
        self._fitted = False

    def _fit_antehoc(self, X, y):
        """
        Fitting the rule based model (antehoc version).

        Args:
            X (pandas.DataFrame): model input data
            y (array): model output data
        """
        self.explainer.fit(X, y)
        self._fitted = True

    def _fit_posthoc(self, X, preprocess_X=None, postprocess_y=None):
        """
        Fitting the rule based model to posthoc interpret another model.

        Args:
            X: input to the model to be interpreted. Type depends on the model.
            preprocess_X: function to create a pandas.DataFrame from the model input to feed to this rule-based model.
            postprocess_y: function to postprocess the model output to feed to this rule-based model.
        """
        y = self._to_interpret.predict(X)

        processed_X = X
        processed_y = y

        if preprocess_X is not None:
            processed_X = preprocess_X(processed_X)

        if postprocess_y is not None:
            processed_y = postprocess_y(processed_y)

        self._fit_antehoc(processed_X, processed_y)

    def interpret(self, explanation_configs={}, path=None):
        """
        Produce explanation.

        Args:
            explanation_configs (dict): keyword arguments for the explain
                function of the explainer. Refer to the AIX360 implementation
                for details.
            path (str): path where to save the explanation. If None, a notebook
                environment will be assumed, and the explanation will be
                visualized.

        Returns:
            pd.DataFrame or dict: the explanation.
        """
        if not self._fitted:
            if self.usage_mode == self.UsageMode.ANTE_HOC:
                self._fit_antehoc(self.X_binarized, self.y)
            else:
                self._fit_posthoc(self.X, self.binarizer.transform)

        self.explanation = self.explainer.explain(**explanation_configs)
        if path is None:
            self._visualize_explanation(self.explanation)
        else:
            self._save_explanation(self.explanation, path)
        return self.explanation

    def _visualize_explanation(self, explanation):
        """
        Helper function to visualize the explanation.
        """
        if isinstance(self.explainer, GLRMExplainer):
            with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None
            ):
                print(explanation)
        elif isinstance(self.explainer, BRCGExplainer):
            # from "https://github.com/IBM/AIX360/blob/master/examples/rbm/breast-cancer-br.ipynb"
            isCNF = 'Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:'
            notCNF = 'Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:'
            print(isCNF if explanation['isCNF'] else notCNF)
            print()
            for rule in explanation['rules']:
                print(f'  - {rule}')

    def _save_explanation(self, explanation, path):
        if isinstance(explanation, DataFrame):
            explanation.to_pickle(path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(explanation, f)

    def predict(self, X, **kwargs):
        self.explainer.predict(X, **kwargs)
