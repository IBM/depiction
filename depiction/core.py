"""Core utilities for depiction."""
from enum import Enum


class Task(Enum):
    """Enum indicating the task performed by a model."""
    CLASSIFICATION = 1
    REGRESSION = 2
