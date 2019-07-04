"""Core utilities for depiction."""
from enum import Enum


class Task(Enum):
    """Enum indicating the task performed by a model."""
    CLASSIFICATION = 1
    REGRESSION = 2


class DataType(Enum):
    """Enum indicating the data type used by a model."""
    TABULAR = 1
    TEXT = 2
