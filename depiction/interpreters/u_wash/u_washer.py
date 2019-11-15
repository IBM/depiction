"""
Concrete interface class for models developed at the University of Washington, 
by Marco Tullio Ribeiro and collaborators.


References:
- "Anchors: High-Precision Model-Agnostic Explanations" (https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
- ""Why Should I Trust You?": Explaining the Predictions of Any Classifier" (https://arxiv.org/pdf/1602.04938.pdf)

"""
import numpy as np
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image import LimeImageExplainer
from anchor.anchor_text import AnchorText
from anchor.anchor_tabular import AnchorTabularExplainer
from anchor.anchor_image import AnchorImage
from matplotlib import pyplot as plt
from collections import defaultdict

from ..base.base_interpreter import BaseInterpreter
from ...core import Task, DataType


def show_image_in_notebook_for_lime(explanation, image, model, labels=None):
    image = np.expand_dims(image, axis=0)
    top_k = 4
    logits = model.predict(image).squeeze()
    label_indexes = np.argsort(model.predict(image).squeeze())[::-1][:top_k]
    figure, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for axis, label_index in zip(
        [column for row in axes for column in row], label_indexes
    ):
        image_to_explain, mask = explanation.get_image_and_mask(label_index)
        axis.imshow(image_to_explain, interpolation=None)
        axis.imshow(mask, cmap='jet', alpha=0.5, interpolation=None)
        axis.set_title(
            f'Explain: {labels[label_index] if labels else label_index} '
            f'({logits[label_index]:.2f})'
        )
        axis.set_xticks([], [])
        axis.set_yticks([], [])


def show_image_in_notebook_for_anchor(explanation, image, model, labels=None):
    # NOTE: get feature mask
    feature_to_alpha = defaultdict(float)
    for index, _, value, _, _ in explanation[1]:
        feature_to_alpha[index] = value
    vectorized_feature_to_alpha = np.vectorize(
        lambda feature: feature_to_alpha[feature]
    )
    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(image)
    axes[0].set_title('Original image')
    axes[1].imshow(
        np.expand_dims(
            vectorized_feature_to_alpha(explanation[0]) * .75 + .25, axis=-1
        ) * image
    )
    axes[1].set_title('Image with exaplanations')
    for axis in axes:
        axis.set_xticks([], [])
        axis.set_yticks([], [])


NOTEBOOK_IMAGE_RENDERERS = {
    'lime': show_image_in_notebook_for_lime,
    'anchors': show_image_in_notebook_for_anchor
}


class UWasher(BaseInterpreter):
    """
    Wash away your doubts about ML models with these interpretability methods
    proposed by researchers from the University of Washington.
    """
    AVAILABLE_INTERPRETERS = {
        'lime':
            {
                DataType.TABULAR: LimeTabularExplainer,
                DataType.TEXT: LimeTextExplainer,
                DataType.IMAGE: LimeImageExplainer
            },
        'anchors':
            {
                DataType.TABULAR: AnchorTabularExplainer,
                DataType.TEXT: AnchorText,
                DataType.IMAGE: AnchorImage
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
            interpreter (str): string denoting the actual model to use.
            Possible values: 'lime', 'anchors'.
            model (base model): task type
            explanation_configs (dict): parameters for the explanation
            kwargs (dict): paramater list to pass to the constructor of the
                explainers. Please, refer to the official implementations of
                LIME and anchors to understand this parameters.
        """
        super(UWasher, self).__init__(model)

        self.model = model
        self.interpreter = interpreter
        Interpreter_model = self.AVAILABLE_INTERPRETERS[interpreter][
            model.data_type]
        if self.model.data_type == DataType.IMAGE:
            self.labels = kwargs.pop('class_names', None)
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
            sample (object): the sample to interpret.
            callback_args (dict): arguments to pass to the model to get a
                callback function.
            explanation_configs (dict): further configurations for the
                explainer.
            path (str): path where to save interpretation results.
                If not provided, show in notebook.
            callback (function): if not provided, the callback function is
                taken directly from the model.
        """
        if callback is None:
            callback = self.model.callback(**callback_args)
        explanation = self.explainer.explain_instance(
            sample, callback, **explanation_configs
        )
        if path is None:
            if self.model.data_type == DataType.IMAGE:
                NOTEBOOK_IMAGE_RENDERERS[self.interpreter](
                    explanation, sample, self.model, self.labels
                )
            else:
                explanation.show_in_notebook()
        elif hasattr(explanation, 'save_to_file'):
            explanation.save_to_file(path)
        return explanation
