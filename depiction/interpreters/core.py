"""Core utilities for handling interpreters."""
from ..core import Task


class Interpreter(object):
    """Abstract implementation of an interpreter."""
    def __init__(self, task, data_type):
        """
        Initalize a Model.
        
        Arguments:
            task (Task): task type.
            args (list): list of arguments. Unused.
            kwargs (dict): list of key-value arguments. Unused.
        """
        self.task = task
        self.data_type = data_type

    def interpret(self, callback, sample):
        """
        Interpret predictions on a sample using a model callback.

        Arguments:
            callback (Callable): a model callback taking a sample as input.
            sample (object): an input sample for the model.

        Returns:
            interpretability results.
        """
        raise NotImplementedError
    
    def interpret_many(self, callback, samples):
        """
        Interpret predictions on the given samples.
        
        Arguments:
            callback (Callable): a model callback taking a sample as input.
            samples (Iterable): input samples for the model.

        Returns:
            a generator of interpretability results.
        """
        for sample in samples:
            yield self.interpret(callback, sample)
