"""Contrastive Explainability Method (without monotonic attribute functions)


References:
    https://arxiv.org/abs/1802.07623
"""
from alibi.explainers import CEM as CEMImplementation

from ....core import DataType, Task
from ...base.base_interpreter import BaseInterpreter


class CEM(BaseInterpreter):
    """Contrastive Explainability Method

    Wrapper for alibis implementation of CEM, which solves the optimization
    problem for finding pertinent positives and negatives using tensorflow.
    """

    SUPPORTED_TASK = {Task.CLASSIFICATION}
    SUPPORTED_DATATYPE = {DataType.TABULAR, DataType.IMAGE}

    def __init__(
        self,
        model,
        mode,
        shape,
        kappa=0.,
        beta=.1,
        feature_range=(-1e10, 1e10),
        gamma=0.,
        ae_model=None,
        learning_rate_init=1e-2,
        max_iterations=1000,
        c_init=10.,
        c_steps=10,
        eps=(1e-3, 1e-3),
        clip=(-100., 100.),
        update_num_grad=1,
        no_info_val=None,
        write_dir=None,
        sess=None
    ):
        """Constructor.

        References:
            CEM implementation, parameter docstrings and defaults adapted from:
            https://github.com/SeldonIO/alibi/blob/92e8048ea2f4e4ef57b6874fa854b90de8ed9602/alibi/explainers/cem.py#L16  # noqa

        The major difference in the constructor signature is that model and
        ae_model shoud be instances of depictions BaseModel
        instead of type Union[Callable, tf.keras.Model, 'keras.Model']

        Args:
            model (BaseModel): Instance implementing predict method returning
                class probabilities that is passed to the explainer
                implementation.
            mode (str): Find pertinent negatives ('PN') or
                pertinent positives ('PP').
            shape (tuple): Shape of input data starting with batch size of 1.
            kappa (float, optional): Confidence parameter for the attack loss
                term. Defaults to 0..
            beta (float, optional): Regularization constant for L1 loss term.
                Defaults to .1.
            feature_range (tuple, optional): Tuple with min and max ranges to
                allow for perturbed instances. Min and max ranges can be floats
                or numpy arrays with dimension (1x nb of features) for
                feature-wise ranges. Defaults to (-1e10, 1e10).
            gamma (float, optional): Regularization constant for optional
                auto-encoder loss term. Defaults to 0..
            ae_model (tf.keras.Model, 'keras.Model', optional): Auto-encoder
                model used for loss regularization. Only keras is supported.
                Defaults to None.
            learning_rate_init (float, optional): Initial learning rate of
                optimizer. Defaults to 1e-2.
            max_iterations (int, optional): Maximum number of iterations for
                finding a PN or PP. Defaults to 1000.
            c_init (float, optional): Initial value to scale the attack loss
                term. Defaults to 10..
            c_steps (int, optional): Number of iterations to adjust the
                constant scaling the attack loss term. Defaults to 10.
            eps (tuple, optional): If numerical gradients are used to compute
                `dL/dx = (dL/dp) * (dp/dx)`, then eps[0] is used to calculate
                `dL/dp` and eps[1] is used for `dp/dx`. eps[0] and eps[1] can
                be a combination of float values and numpy arrays. For eps[0],
                the array dimension should be (1x nb of prediction categories)
                and for eps[1] it should be (1x nb of features).
                Defaults to (1e-3, 1e-3).
            clip (tuple, optional): Tuple with min and max clip ranges for both
                the numerical gradients and the gradients obtained from the
                TensorFlow graph. Defaults to (-100., 100.).
            update_num_grad (int, optional): If numerical gradients are used,
                they will be updated every update_num_grad iterations.
                Defaults to 1.
            no_info_val (Union[float, np.ndarray], optional): Global or
                feature-wise value considered as containing no information.
                Defaults to None, in this case fit method needs to be called.
            write_dir (str, optional): Directory to write tensorboard files to.
                Defaults to None.
            sess (tf.compat.v1.Session, optional): Optional Tensorflow session
                that will be used if passed instead of creating or inferring
                one internally. Defaults to None.
        """
        super().__init__(model)

        self.explainer = CEMImplementation(
            model.predict, mode, shape, kappa, beta, feature_range, gamma,
            ae_model,
            learning_rate_init, max_iterations, c_init, c_steps, eps, clip,
            update_num_grad, no_info_val, write_dir, sess
        )

    def interpret(self, X, Y=None, verbose=False):
        """Explain instance and return PP or PN with metadata.

        Args:
            X (np.ndarray): Instances to attack.
            Y (np.ndarray, optional): Labels for X.
            verbose(bool, optinal): Print intermediate results of optimization

        Returns:
            dict: the PP or PN with additional metadata
        """
        return self.explainer.explain(X, Y, verbose)

    def fit(self, train_data, no_info_type='median'):
        """Get 'no information' values from the training data.

        Args:
          train_data (np.ndarray): Representative sample from the training
            data.
          no_info_type (str, optional): 'median' or 'mean' value by feature
            supported. Defaults to 'median'.
        """
        self.explainer.fit()
