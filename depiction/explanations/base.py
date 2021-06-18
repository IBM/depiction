from abc import ABC, abstractmethod


class BaseExplanation(ABC):
    """
    Interface class for explanations
    """
    def __init__(self):
        pass

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError


