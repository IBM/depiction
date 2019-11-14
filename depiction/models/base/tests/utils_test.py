import unittest
from unittest import mock

from depiction.models.base.utils import MODELS_SUBDIR, get_model_file


class BaseUtilsTestCase(unittest.TestCase):

    def testGetModelFile(self):
        fname = 'test_file.h5'
        origin = 'test_url'
        cache_dir = 'cache_path'

        with mock.patch(
            'depiction.models.base.utils.get_file'
        ) as mock_get_file:
            get_model_file(fname, origin, cache_dir)

            mock_get_file.assert_called_with(
                fname, origin, cache_subdir=MODELS_SUBDIR, cache_dir=cache_dir
            )


if __name__ == "__main__":
    unittest.main()
