"""
Depiction wrappers around SKLearn linear models
"""
from depiction.core import Task, DataType
from depiction.interpreters.base.base_interpreter import AnteHocInterpreter
from sklearn.linear_model import LinearRegression, LogisticRegression

import numpy as np
from depiction.explanations.base import BaseExplanation
from depiction.explanations.feature_attribution import FeatureAttributionExplanation
from copy import deepcopy
from types import GeneratorType


class LinearModel(AnteHocInterpreter):
    SUPPORTED_TASK = {
        Task.CLASSIFICATION,
        Task.REGRESSION
    }
    SUPPORTED_DATATYPE = {
        DataType.TABULAR
    }
    """
    Wrapper to SKLearn Linear models. 
    If task is regression, this model is a LinearRegression
    If task is classification, this model is LogisticRegression
    """
    def __init__(self, usage_mode, model=None, task_type=None, data_type=None, linear_kwargs={}):
        """
        Constructor.

        Args:
            usage_mode: either AnteHocInterpret.UsageMode.ANTE_HOC or AnteHocInterpret.UsageMode.POST_HOC. Specifies how
                        this antehoc model will be used (i.e. directly on data, or to post-hoc interpret another model)
            model (BaseModel): if in post_hoc mode, a model to interpret should be provided
            task_type (depiction.core.Task): task type. To define, for ante-hoc mode
            data_type (depiction.core.Task): task type. To define, for ante-hoc mode
            linear_kwargs (dict): arguments to construct the linear models.
                                Please refer to the SKLearn documentation, for info about the arguments.
        """
        super(LinearModel, self).__init__(usage_mode, model, task_type, data_type)
        if self.task == Task.CLASSIFICATION:
            self._model = LogisticRegression(**linear_kwargs)
        else:
            self._model = LinearRegression(**linear_kwargs)

    def _fit_antehoc(self, X, y, additional_args={}):
        """
        Fitting function. Takes the same arguments as the SkLearn DecisionTrees. Data is given through the arguments
        X and y. while additional arguments are passed via the dict additional args.
        """
        self._model.fit(X, y, **additional_args)

    def _fit_posthoc(self, X, additional_args={}):
        y = self._to_interpret.predict(X)
        if isinstance(y, GeneratorType):
            y = list(y)
            y = np.concatenate(y, axis=0)
        self._model.fit(X, y, **additional_args)

    def interpret(self):
        return LinearModelExplanation(self)

    def _predict(self, sample, *args, **kwargs):
        return self._model.predict(sample)


class LinearModelExplanation(BaseExplanation):
    """
    Explanation class for linear models
    """
    def __init__(self, model: LinearModel):
        self._model = model

    def visualize(self, feature_names=None, output_names=None, ordered=False, top_k=None,**kwargs):
        if feature_names is None:
            feature_names = ['Feature_{}'.format(i) for i in range(self._model._model.coef_.shape[1])]
        if output_names is None:
            if self._model.task == Task.CLASSIFICATION:
                base_str = 'Class'
            else:
                base_str = 'Output'
            output_names = [base_str + '_{}'.format(i) for i in range(self._model._model.coef_.shape[0])]
        if top_k is None:
            top_k = len(feature_names)

        model_report = ''
        if self._model.task is Task.CLASSIFICATION:
            pre_str = 'Logit'
        else:
            pre_str = 'Value'
        for i, o in enumerate(output_names):
            model_report += pre_str + ' {}  is computed as:\n'.format(o)
            order = np.arange(len(feature_names))
            if ordered:
                order = sorted(order, key=lambda x: np.abs(self._model._model.coef_[i, x]), reverse=True)
            for j, idx in enumerate(order[:top_k]):
                f = feature_names[idx]
                model_report += '\t{:.2f} * {}'.format(self._model._model.coef_[i, idx], f)
                if j == len(feature_names) - 1:
                    model_report += '\n\n'
                else:
                    model_report += ' +\n'
        print(''.join(model_report))

    def to_feature_attribution(self, sample: np.ndarray=None):
        if sample is not None:
            feat_attr = self._model._model.coef_*sample
        else:
            feat_attr = self._model._model.coef_
        return FeatureAttributionExplanation(feat_attr, self._model.data_type)
