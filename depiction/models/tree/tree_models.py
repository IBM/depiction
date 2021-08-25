"""
Wrappers for tree models
"""
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from depiction.interpreters.base.base_interpreter import AnteHocInterpreter
from depiction.core import Task, DataType
import numpy as np
from depiction.explanations.base import BaseExplanation
from matplotlib import pyplot as plt


class TreeModel:
    """
    This class will be used only as a type trait. So no particular implementation is needed.
    """
    pass


class DecisionTree(AnteHocInterpreter, TreeModel):
    SUPPORTED_TASK = {
        Task.CLASSIFICATION,
        Task.REGRESSION
    }
    SUPPORTED_DATATYPE = {
        DataType.TABULAR
    }
    """
    Wrapper for simple decision trees
    """
    def __init__(self, usage_mode, model=None, task_type=None, data_type=None, tree_kwargs={}):
        """
        Constructor.

        Args:
            usage_mode: either AnteHocInterpret.UsageMode.ANTE_HOC or AnteHocInterpret.UsageMode.POST_HOC. Specifies how
                        this antehoc model will be used (i.e. directly on data, or to post-hoc interpret another model)
            model (BaseModel): if in post_hoc mode, a model to interpret should be provided
            task_type (depiction.core.Task): task type. To define, for ante-hoc mode
            data_type (depiction.core.Task): task type. To define, for ante-hoc mode
            tree_kwargs (dict): arguments to construct the tree. Please refer to the SKLearn DecisionTree documentation,
                                for info about the arguments.
        """
        super(DecisionTree, self).__init__(usage_mode, model, task_type, data_type)
        if self.task == Task.CLASSIFICATION:
            self._model = DecisionTreeClassifier(**tree_kwargs)
        else:
            self._model = DecisionTreeRegressor(**tree_kwargs)

    def _fit_antehoc(self, X, y, additional_args={}):
        """
        Fitting function. Takes the same arguments as the SkLearn DecisionTrees. Data is given through the arguments
        X and y. while additional arguments are passed via the dict additional args.
        """
        self._model.fit(X, y, **additional_args)

    def _fit_posthoc(self, X, additional_args={}):
        y = np.array([self._to_interpret.predict_many(X)])
        self._model.fit(X, y, **additional_args)

    def interpret(self):
        return DecisionTreeExplanation(self)


class DecisionTreeExplanation(BaseExplanation):
    def __init__(self, tree_model: DecisionTree):
        super(DecisionTreeExplanation, self).__init__()
        self._model = tree_model

    def visualize(self, fig_axes=None, show=True, kwargs={}):
        """
        Visualization routine.

        Args:
            fig_axes: tuple containing matplotlib fig and axis objects, used for plotting.
                      If not provided, a new plot will be generated.
            show (bool): if True, show the plot
            kwargs (dict): keyword arguments to pass to the tree plotting function. Please refer to
        """
        if fig_axes is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_axes

        kwargs['ax'] = ax
        plot_tree(self._model._model, **kwargs)
        if show:
            plt.show()

    def to_rule_explanation(self):
        """
        Function to cast this explanation to a textual rule
        """
        raise NotImplementedError
