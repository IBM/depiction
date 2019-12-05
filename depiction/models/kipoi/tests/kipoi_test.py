"""Test KipoiModel."""
import unittest

import numpy as np
from concise.preprocessing.sequence import encodeDNA
from depiction.core import DataType, Task
from depiction.models.kipoi.core import KipoiModel


def preprocessing_function(nucleotide_sequence: str) -> np.ndarray:
    """One-hot-encode (a single) the sequence.

    The kipoi deepbind model does not accept a string of nucleotides.

    Args:
        nucleotide_sequence (str): defined to be of lenght 101, though other
            lenghts might be accepted.

    Returns:
        np.ndarray: of shape `[1, len(nucleotide_sequence), 4]`
    """
    return encodeDNA([nucleotide_sequence])


class KipoiModelTestCase(unittest.TestCase):
    """Test KipoiModel.

    Kopoi model page:
    http://kipoi.org/models/DeepBind/Homo_sapiens/TF/D00817.001_ChIP-seq_TBP/
    """

    def test_prediction(self):
        model = KipoiModel(
            'DeepBind/Homo_sapiens/TF/D00817.001_ChIP-seq_TBP',
            Task.CLASSIFICATION, DataType.TEXT,
            preprocessing_function=preprocessing_function
        )
        sequence = 'ATGGGCCAGCACACAGACCAGCACGTTGCCCAGGAGCTCGCTATAAAAGGGCGTGGGAGGAAGATAAGAGGTATGAACATGATTAGCAAAAGGGCCTAGCT'  # noqa
        # contains the TATA box:                             ~~~~~~~
        self.assertTrue((model.predict(sequence) > 0)[0])  # shape is (1,)


if __name__ == "__main__":
    unittest.main()
