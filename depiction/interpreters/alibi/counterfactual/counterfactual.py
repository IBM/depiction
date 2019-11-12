"""Counterfactual explanation method based on Wachter et al. (2017)


References:
    https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf
"""
from alibi.explainers import CounterFactual

from ....core import DataType, Task
from ...base.base_interpreter import BaseInterpreter


class Counterfactual(BaseInterpreter):
    """Counterfactual explanation.

    Wrapper for alibis implementation of counterfactual exaplanation.
    """

    SUPPORTED_TASK = {Task.CLASSIFICATION}
    SUPPORTED_DATATYPE = {DataType.TABULAR, DataType.IMAGE}

    def __init__(
        self,
        model,
        shape,
        distance_fn='l1',
        target_proba=1.0,
        target_class='other',
        max_iter=1000,
        early_stop=50,
        lam_init=1e-1,
        max_lam_steps=10,
        tol=0.05,
        learning_rate_init=0.1,
        feature_range=(-1e10, 1e10),
        eps=0.01,  # feature-wise epsilons
        init='identity',
        decay=True,
        write_dir=None,
        debug=False,
        sess=None
    ):
        """Constructor.

        References:
            Counterfactual explanation implementation, parameter docstrings and defaults adapted from:
            https://github.com/SeldonIO/alibi/blob/14804f07457da881a5f70ccff2dcbfed2378b860/alibi/explainers/counterfactual.py#L85  # noqa

        The major difference in the constructor signature is that model and
        ae_model shoud be instances of depictions BaseModel
        instead of type Union[Callable, tf.keras.Model, 'keras.Model']

        Args:
            model (BaseModel): Instance implementing predict method returning
                class probabilities that is passed to the explainer
                implementation.
            shape (tuple): Shape of input data starting with batch size.
            distance_fn (str, optional): Distance function to use in the loss term. Defaults to 'l1'.
            target_proba (float, optional): Target probability for the counterfactual to reach. Defaults to 1.0.
            target_class (Union[str, int], optional): Target class for the counterfactual to reach, one of 'other',
                'same' or an integer denoting desired class membership for the counterfactual instance. Defaults to 'other'.
            max_iter (int, optional): Maximum number of interations to run the gradient descent for (inner loop).
                Defaults to 1000.
            early_stop (int, optional): Number of steps after which to terminate gradient descent if all or none of found
                instances are solutions. Defaults to 50.
            lam_init (float, optional): Initial regularization constant for the prediction part of the Wachter loss.
                Defaults to 1e-1.
            max_lam_steps (int, optional): Maximum number of times to adjust the regularization constant (outer loop)
                before terminating the search. Defaults to 10.
            tol (float, optional): Tolerance for the counterfactual target probability. Defaults to 0.05.
            learning_rate_init (float, optional): Initial learning rate for each outer loop of lambda. Defaults to 0.1.
            feature_range (Union[Tuple, str], optional): Tuple with min and max ranges to allow for perturbed instances.
                Min and max ranges can be floats or numpy arrays with dimension (1 x nb of features)
                for feature-wise ranges. Defaults to (-1e10, 1e10).
            eps (Union[float, np.ndarray], optional): Gradient step sizes used in calculating numerical gradients,
                defaults to a single value for all features, but can be passed an array for
                feature-wise step sizes. Defaults to 0.01.
            init (str): Initialization method for the search of counterfactuals, currently must be 'identity'.
            decay (bool, optional): Flag to decay learning rate to zero for each outer loop over lambda.
                Defaults to True.
            write_dir (str, optional): Directory to write Tensorboard files to. Defaults to None.
            debug (bool, optional): Flag to write Tensorboard summaries for debugging. Defaults to False.
            sess (tf.compat.v1.Session, optional): Optional Tensorflow session that will be used if passed
                instead of creating or inferring one internally. Defaults to None.
        """
        super().__init__(model)

        self.explainer = CounterFactual(
            model.predict, shape, distance_fn, target_proba, target_class,
            max_iter, early_stop, lam_init, max_lam_steps, tol,
            learning_rate_init, feature_range, eps, init, decay, write_dir,
            debug, sess
        )

    def interpret(self, X):
        """
        Explain an instance and return the counterfactual with metadata.

        Args:
            X (np.ndarray): Instance to be explained.

        Returns:
            dict: a dictionary containing the counterfactual
                with additional metadata.
        """
        return self.explainer.explain(X)

    def fit(self, X=None, y=None):
        """
        Since the interpreter is unsupervised the method
        is not doing anything.

        Args:
            X (np.ndarray): training data. Defaults to None.
            y (np.ndarray): optional labels. Defaults to None.
        """
        self.explainer.fit(X, y)
