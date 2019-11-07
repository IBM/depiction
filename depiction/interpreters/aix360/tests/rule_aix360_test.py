import unittest
from unittest import mock

from ....models.base.base_model import BaseModel
from ..rule_based_model import RuleAIX360


class DummyModel(BaseModel):
    def predict(self, sample):
        return sample


class RuleAIX360TestCase(unittest.TestCase):
    def testConstructor(self):
        # test error for wrong model
        