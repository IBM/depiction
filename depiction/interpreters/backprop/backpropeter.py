"""Backpropagation-like methods for interpretability

Wrapper around:
- (pytorch) Captum [1]
- (keras) DeepExplain [2]

References:
    [1] https://captum.ai/
    [2] https://arxiv.org/abs/1711.06104
"""
from captum.attr import IntegratedGradients, Saliency, DeepLift,\
                        DeepLiftShap, GradientShap, InputXGradient 
from tensorflow.keras import backend as K
from deepexplain.tensorflow import DeepExplain
from tensorflow.keras.models import Model
from deepexplain.tensorflow.methods import attribution_methods
from copy import deepcopy
import warnings
from captum.attr import visualization as viz
import numpy as np
from matplotlib.colors import Normalize
from depiction.explanations.feature_attribution import FeatureAttributionExplanation

from ...core import DataType, Task
from ..base.base_interpreter import BaseInterpreter
from depiction.models.torch.core import TorchModel
from depiction.models.keras.core import KerasModel


def _preprocess_att_methods_keras():
    methods = deepcopy(attribution_methods)
    methods.pop('deeplift')
    return methods


class BackPropeter(BaseInterpreter):
    """Backpropagation-like Explainability Method

    Wrapper for Captum and DeepExplain implementations.
    """
    SUPPORTED_TASK = {Task.CLASSIFICATION}
    SUPPORTED_DATATYPE = {DataType.TABULAR, DataType.IMAGE, DataType.TEXT}


    METHODS = {
        'torch': {
            'integrated_grads': IntegratedGradients,
            'saliency': Saliency,
            'deeplift': DeepLift,
            'deeplift_shap': DeepLiftShap,
            'gradient_shap': GradientShap,
            'inputxgrad': InputXGradient
        },
        'keras': _preprocess_att_methods_keras()
    }

    @classmethod
    def _check_supported_method(self, model_type, method):
        if method not in self.METHODS[model_type]:
            raise ValueError('Method {} not supported! At the moment we only support: {}.'.format(
                                method,self.METHODS[model_type].keys()))

    def __init__(self, model, method, **method_kwargs):
        """
        Constructor for backpropagation-like methods.

        Reference:
            https://captum.ai/api/attribution.html

        Args:
            model (TorchModel or KerasModel): model to explain
            method (str): method to use
            method_kwargs: keyword args to pass on to the explainer constrcutor.
                           Please refer to the the specific algorithm (following the above link)
                           to see and understand the available arguments.
        """
        super(BackPropeter, self).__init__(model)

        self._model = model
        self._method = method

        if isinstance(self._model, TorchModel):
            self._check_supported_method('torch', method)
            self._explainer = self.METHODS['torch'][method](self._model._model, **method_kwargs)
        elif isinstance(self._model, KerasModel):
            self._check_supported_method('keras', method)
        else:
            raise ValueError('Model not supported! At the moment we only support {}.'
                             '\nPlease check again in the future!'.format(self.METHODS.keys()))

    def interpret(self, samples, target_layer=-1, explanation_configs={}):
        """Explain instance and return PP or PN with metadata. If pyTorch (captum) is used,
        the convergence delta is NOT returned by default.

        Args:
            samples (tensor or tuple of tensors): Samples to explain
            target_layer (int): for KerasModel, specify the target layer. 
                                Following example in: https://github.com/marcoancona/DeepExplain/blob/master/examples/mint_cnn_keras.ipynb
            explanation_configs (dict): optional arguments to pass to the explainer for attribution. optional.

        Returns:
            tensor (or tuple of tensors) containing attributions
        """
        if isinstance(self._model, TorchModel):
            if self._explainer.has_convergence_delta() and 'return_convergence_delta' not in explanation_configs:
                explanation_configs['return_convergence_delta'] = False
            explanations = self._explainer.attribute(inputs=self._model._prepare_sample(samples), **explanation_configs)

            def post_process(exp):
                if self._model.data_type == DataType.IMAGE:
                    return exp.transpose(0, 1).transpose(1, 2)
                else:
                    return exp
            explanations = [post_process(exp).detach().cpu().numpy() for exp in explanations]

        else:
            warnings.warn("This branch of code may be deprecated!")
            with DeepExplain(session=K.get_session()) as de:
                input_tensor = self._model._model.inputs
                smpls = samples if isinstance(samples, list) else [samples]
                if self._method in {'occlusion', 'shapley_sampling'}:
                    warnings.warn('For perturbation methods, multiple inputs (modalities) are not supported.', UserWarning)
                    smpls = smpls[0]
                    input_tensor = input_tensor[0]

                model = Model(inputs=input_tensor, outputs=self._model._model.outputs)
                target_tensor = model(input_tensor)
                explanations = de.explain(self._method, T=target_tensor, X=input_tensor, xs=smpls, **explanation_configs)
                if self._method in {'occlusion', 'shapley_sampling'}:
                    explanations = np.expand_dims(explanations, axis=0)

        return [FeatureAttributionExplanation(exp, self._model.data_type) for exp in explanations]
