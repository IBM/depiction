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
from skimage.color import gray2rgb
from matplotlib.colors import Normalize

from ..base.base_interpreter import BaseInterpreter
from ...core import Task, DataType


def show_image_in_notebook_for_lime(explanation, image, callback, labels=None, top_k=4, nperrow=2):
    """
    Show in notebook for LIME images.

    Args:
        explanation (object): LIME explanation.
        image (np.ndarray): array representing the image.
        callback (function): model callback function. 
        labels (list, optional): label names. Defaults to None.
        top_k (int, optional): number of top classes to show.
        n_row (int, optional): number of explanations to show per row.
    """
    image = np.expand_dims(image, axis=0)
    logits = callback(image).squeeze()
    label_indexes = np.argsort(callback(image).squeeze())[::-1][:top_k]
    top_k = min(top_k, len(label_indexes))
    rows = np.int(np.ceil(np.float(top_k)/nperrow))
    figure, axes = plt.subplots(rows, nperrow, sharex=True, sharey=True)
    axes = axes.flatten()
    for axis, label_index in zip(
        axes, label_indexes
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


def show_image_in_notebook_for_anchor(
    explanation, image, callback, labels=None
):
    """
    Show in notebook for Anchors images.

    Args:
        explanation (object): Anchors explanation.
        image (np.ndarray): array representing the image.
        callback (function): model callback function.
        labels (list, optional): label names. Defaults to None.
    """
    # NOTE: get feature mask
    feature_to_alpha = defaultdict(float)
    for index, _, value, _, _ in explanation[1]:
        feature_to_alpha[index] = value
    vectorized_feature_to_alpha = np.vectorize(
        lambda feature: feature_to_alpha[feature]
    )
    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    normalizer = Normalize()
    img = normalizer(image)
    axes[0].imshow(img)
    axes[0].set_title('Original image')
    axes[1].imshow(
        np.expand_dims(
            vectorized_feature_to_alpha(explanation[0]) * .75 + .25, axis=-1
        ) * img
    )
    axes[1].set_title('Image with explanations')
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

    def __init__(
        self, interpreter, model,
        train_data=None, train_labels=None,
        validation_data=None, validation_labels=None,
        discretizer='quartile', **kwargs
    ):
        """
        Constructor.

        Args:
            interpreter (str): string denoting the actual model to use.
                Possible values: 'lime', 'anchors'.
            model (base model): task type.
            explanation_configs (dict): parameters for the explanation.
            kwargs (dict): paramater list to pass to the constructor of the
                explainers. Please, refer to the official implementations of
                LIME and anchors to understand this parameters.

        In the special case of 'anchor' and model.data_type.TABULAR,
        additionally required arguments (for fitting discretizer) are:
            train_data (np.ndarray)
            train_labels (np.ndarray)
            validation_data (np.ndarray)
            validation_labels (np.ndarray)
            discretizer (str, optional): 'quartile' or 'decile'
        """

        super(UWasher, self).__init__(model)

        self.model = model
        self.interpreter = interpreter
        Interpreter_model = self.AVAILABLE_INTERPRETERS[interpreter][
            model.data_type]
        self.image_data = self.model.data_type == DataType.IMAGE
        if self.image_data:
            self.labels = kwargs.pop('class_names', None)
        self.explainer = Interpreter_model(**kwargs)

        if (
            self.interpreter == 'anchors'
            and self.model.data_type is DataType.TABULAR
        ):
            if (
                train_data is None
                or train_labels is None
                or validation_data is None
                or validation_labels is None
            ):
                raise TypeError(
                    "In the special case of 'anchor' and "
                    "model.data_type.TABULAR, additionally required arguments "
                    "(for fitting discretizer) are: train_data, train_labels, "
                    "validation_data, validation_labels"
                )
            self.explainer.fit(
                train_data, train_labels,
                validation_data, validation_labels, discretizer
            )

    def interpret(
        self,
        sample,
        callback_args={},
        explanation_configs={},
        vis_configs={},
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
            if self.image_data:
                if len(sample.shape) == 4:
                    if sample.shape[0] != 1:
                        raise RuntimeError(
                            'Can explain only single examples. '
                            'Make share the batch size is 1. '
                            f'Current shape: {sample.shape}'
                        )
                    else:
                        sample = sample[0, ...]
                if sample.shape[-1] == 1:
                    sample = gray2rgb(sample.squeeze())
                    # NOTE: avoid recursion errors
                    _callback = callback
                    del (callback)
                    callback = lambda sample: _callback(  # noqa
                        np.expand_dims(sample[..., -1], axis=-1)
                    )
        explanation = self.explainer.explain_instance(
            sample, callback, **explanation_configs
        )
        if path is None:
            if self.image_data:
                NOTEBOOK_IMAGE_RENDERERS[self.interpreter](
                    explanation, sample, callback, self.labels, **vis_configs
                )
            else:
                explanation.show_in_notebook()
        elif hasattr(explanation, 'save_to_file'):
            explanation.save_to_file(path)
        return explanation
