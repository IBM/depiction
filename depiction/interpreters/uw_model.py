"""
Concrete interface class for models developed at the University of Washington, 
by Marco Tullio Ribeiro and collaborators.


References:
- "Anchors: High-Precision Model-Agnostic Explanations" (https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
- ""Why Should I Trust You?": Explaining the Predictions of Any Classifier" (https://arxiv.org/pdf/1602.04938.pdf)

""" 
from .core import Interpreter
from ..core import DataType
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from anchor.anchor_tabular import AnchorTabularExplainer
from anchor.anchor_text import AnchorText


AVAILABLE_MODELS = {
    'lime': {
        DataType.TABULAR: LimeTabularExplainer,
        DataType.TEXT: LimeTextExplainer
    },
    'anchor': {
        DataType.TABULAR: AnchorTabularExplainer,
        DataType.TEXT: AnchorText
    }
}


class UWModel(Interpreter):
    """
    Concrete interface class
    """
    def __init__(self, interpreter, task, data_type, explanation_configs, **kwargs):
        """
        Constructor.

        Arguments:
            task (Task): task type
            explanation_configs (dict): parameters for the explanation
            kwargs (dict): paramater list to pass to the constructor of the explainers.
                      Please, refer to the official implementations of LIME and anchors to
                      understand this parameters.
        """
        super(UWModel, self).__init__(task, data_type)

        self.explanation_configs = explanation_configs

        Model = AVAILABLE_MODELS[interpreter][data_type]
        self.explainer = Model(**kwargs)


    def interpret(self, callback, sample, path=None):
        explanation = self.explainer.explain_instance(sample, callback, **self.explanation_configs)
        if path is None:
            explanation.show_in_notebook()
        else:
            explanation.save_to_file(path)