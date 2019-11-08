"""Test MAX toxic comment classifier."""
import os
import unittest

from ..toxic_comment_classifier import ToxicCommentClassifier


class ToxicCommentClassifierTestCase(unittest.TestCase):
    """Test MAX toxic comment classifier."""

    def test_prediction(self):
        toxic = ToxicCommentClassifier(
            uri='http://{os.environ.get("TEST_MAX_HOST", "localhost")}:5000'
        )
        texts = ['This movie sucks.', 'I really liked the play.']
        self.assertEqual(
            toxic.predict(texts).shape, (len(texts), len(toxic.labels))
        )
