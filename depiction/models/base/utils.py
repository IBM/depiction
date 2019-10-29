"""Generic util functions for models."""
from tensorflow.keras.utils import get_file


def get_model_file(filename, origin, cache_dir):
    """
    Downloads a file from a URL if it not already in the cache.
    """
    return get_file(filename,
                    origin,
                    cache_subdir='models',
                    cache_dir=cache_dir)
