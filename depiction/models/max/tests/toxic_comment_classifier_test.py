"""Test MAX toxic comment classifier."""
import os
import unittest

from depiction.models.max.toxic_comment_classifier import \
    ToxicCommentClassifier


class ToxicCommentClassifierTestCase(unittest.TestCase):
    """Test MAX toxic comment classifier."""

    def test_prediction(self):
        toxic = ToxicCommentClassifier(
            uri='http://{}:5000'.format(
                os.environ.
                get('TEST_MAX_TOXIC_COMMENT_CLASSIFIER', 'localhost')
            )
        )
        texts = ['This movie sucks.', 'I really liked the play.']
        self.assertEqual(
            toxic.predict(texts).shape, (len(texts), len(toxic.labels))
        )


if __name__ == "__main__":
    unittest.main()
