from abc import ABC, abstractmethod


class BaseExplanation(ABC):
    """
    Interface class for explanations
    """
    def __init__(self):
        pass

    @abstractmethod
    def visualize(self, *args, **kwargs):
        """
        Main function for explanations. Produces a visualization of the explanations.
        All interpretable methods should return a subclass of base explanation.
        """
        raise NotImplementedError


