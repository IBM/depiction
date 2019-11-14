import unittest

from depiction.core import Task


class TaskTestCase(unittest.TestCase):

    def testLtOperator(self):
        self.assertLess(Task.BINARY, Task.CLASSIFICATION)
        self.assertLess(Task.MULTICLASS, Task.CLASSIFICATION)

        self.assertFalse(Task.CLASSIFICATION < Task.BINARY)
        self.assertFalse(Task.CLASSIFICATION < Task.MULTICLASS)

        self.assertFalse(Task.CLASSIFICATION < Task.REGRESSION)
        self.assertFalse(Task.REGRESSION < Task.CLASSIFICATION)

        self.assertFalse(Task.BINARY < Task.BINARY)

    def testLeOperator(self):
        self.assertLessEqual(Task.BINARY, Task.CLASSIFICATION)
        self.assertLessEqual(Task.MULTICLASS, Task.CLASSIFICATION)
        self.assertLessEqual(Task.BINARY, Task.BINARY)

        self.assertFalse(Task.CLASSIFICATION <= Task.REGRESSION)
        self.assertFalse(Task.REGRESSION <= Task.CLASSIFICATION)

    def testGtOperator(self):
        self.assertGreater(Task.CLASSIFICATION, Task.BINARY)
        self.assertGreater(Task.CLASSIFICATION, Task.MULTICLASS)

        self.assertFalse(Task.BINARY > Task.CLASSIFICATION)
        self.assertFalse(Task.MULTICLASS > Task.CLASSIFICATION)

        self.assertFalse(Task.CLASSIFICATION > Task.REGRESSION)
        self.assertFalse(Task.REGRESSION > Task.CLASSIFICATION)

        self.assertFalse(Task.BINARY > Task.BINARY)

    def testGeOperator(self):
        self.assertGreaterEqual(Task.CLASSIFICATION, Task.BINARY)
        self.assertGreaterEqual(Task.CLASSIFICATION, Task.MULTICLASS)
        self.assertGreaterEqual(Task.BINARY, Task.BINARY)

        self.assertFalse(Task.CLASSIFICATION >= Task.REGRESSION)
        self.assertFalse(Task.REGRESSION >= Task.CLASSIFICATION)

    def testCheckSupport(self):
        supported = [Task.CLASSIFICATION, Task.MULTICLASS]

        self.assertTrue(Task.check_support(Task.BINARY, supported))
        self.assertTrue(Task.check_support(Task.CLASSIFICATION, supported))
        self.assertFalse(Task.check_support(Task.REGRESSION, supported))


if __name__ == "__main__":
    unittest.main()
