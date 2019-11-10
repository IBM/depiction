"""Core utilities for depiction."""
from enum import Enum, Flag, auto


class Task(Flag):
    """Enum indicating the task performed by a model."""
    BINARY = auto()
    MULTICLASS = auto()
    REGRESSION = auto()
    CLASSIFICATION = BINARY | MULTICLASS

    def __lt__(self, other):
        res = (self.value & other.value)
        return (res == self.value) and (res != other.value)

    def __le__(self, other):
        return self.__lt__(other) or (self.value == other.value)

    def __gt__(self, other):
        return ((self.value
                 | other.value) == self.value) and (self.value != other.value)

    def __ge__(self, other):
        return self.__gt__(other) or (self.value == other.value)

    @staticmethod
    def check_support(t, tasks_set):
        """
        Given an iterable containing tasks, checks if 'self' <= to any of the
        tasks in the iterable.

        Args:
            tasks_set (iterable): iterable containing tasks
        """
        for task in tasks_set:
            if t <= task:
                return True
        return False


class DataType(Enum):
    """Enum indicating the data type used by a model."""
    TABULAR = 1
    TEXT = 2
    IMAGE = 3
