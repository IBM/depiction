"""Test MAX toxic comment classifier."""
import os
import unittest

import numpy as np

from depiction.models.max.breast_cancer_mitosis_detector import \
    BreastCancerMitosisDetector


class BreastCancerMitosisDetectorTestCase(unittest.TestCase):
    """Test MAX toxic comment classifier."""

    def test_prediction(self):
        model = BreastCancerMitosisDetector(
            uri='http://{}:5000'.format(
                os.environ.
                get('TEST_MAX_BREAST_CANCER_MITOSIS_DETECTOR', 'localhost')
            )
        )
        image = np.random.randn(64, 64, 3)
        self.assertEqual(model.predict(image).shape, (1, len(model.labels)))


if __name__ == "__main__":
    unittest.main()
