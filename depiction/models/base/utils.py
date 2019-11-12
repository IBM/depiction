"""Generic util functions for models."""
from tensorflow.keras.utils import get_file

MODELS_SUBDIR = 'models'


def get_model_file(filename, origin, cache_dir):
    """
    Downloads a file from a URL if it not already in the cache.
    """
    return get_file(
        filename, origin, cache_subdir=MODELS_SUBDIR, cache_dir=cache_dir
    )
