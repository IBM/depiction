"""
Concrete interface class for models developed at the University of Washington, 
by Marco Tullio Ribeiro and collaborators.


References:
- "Anchors: High-Precision Model-Agnostic Explanations" (https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
- ""Why Should I Trust You?": Explaining the Predictions of Any Classifier" (https://arxiv.org/pdf/1602.04938.pdf)

"""
from anchor.anchor_text import AnchorText
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
from anchor.anchor_tabular import AnchorTabularExplainer

from ..base.base_interpreter import BaseInterpreter
from ...core import Task, DataType


class UWasher(BaseInterpreter):
    """
    Wash away your doubts about ML models with these interpretability methods proposed
    by researchers from the University of Washington.
    """
    AVAILABLE_INTERPRETERS = {
        'lime':
            {
                DataType.TABULAR: LimeTabularExplainer,
                DataType.TEXT: LimeTextExplainer
            },
        'anchors':
            {
                DataType.TABULAR: AnchorTabularExplainer,
                DataType.TEXT: AnchorText
            }
    }

    SUPPORTED_TASK = {Task.CLASSIFICATION}
    SUPPORTED_DATATYPE = {
        k
        for i in AVAILABLE_INTERPRETERS.values() for k in i.keys()
    }

    def __init__(self, interpreter, model, **kwargs):
        """
        Constructor.

        Args:
            interpreter (str): string denoting the actual model to use. Possible values: 'lime', 'anchors'.
            model (base model): task type
            explanation_configs (dict): parameters for the explanation
            kwargs (dict): paramater list to pass to the constructor of the explainers.
                      Please, refer to the official implementations of LIME and anchors to
                      understand this parameters.
        """
        super(UWasher, self).__init__(model)

        self.model = model
        Interpreter_model = self.AVAILABLE_INTERPRETERS[interpreter][
            model.data_type]
        self.explainer = Interpreter_model(**kwargs)

    def interpret(
        self,
        sample,
        callback_args={},
        explanation_configs={},
        path=None,
        callback=None
    ):
        """
        Interpret a sample.

        Args:
            sample: the sample to interpret
            callback_args (dict): arguments to pass to the model to get a callback function
            explanation_configs (dict): further configurations for the explainer
            path (str): path where to save interpretation results. If not provided, show in notebook
            callback: function. if not provided, the callback function is taken directly from the model.
        """
        if callback is None:
            callback = self.model.callback(**callback_args)
        explanation = self.explainer.explain_instance(
            sample, callback, **explanation_configs
        )
        if path is None:
            explanation.show_in_notebook()
        else:
            explanation.save_to_file(path)
        return explanation