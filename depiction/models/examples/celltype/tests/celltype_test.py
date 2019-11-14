import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from depiction.models.examples.celltype.celltype import CellTyper


class CellTyperTestCase(unittest.TestCase):
    """Test celltype classifier."""

    def setUp(self):
        """Prepare data to predict."""
        filepath = Path(__file__).resolve(
        ).parents[5] / 'data' / 'single-cell' / 'data.csv'
        data_df = pd.read_csv(filepath)
        self.data = data_df.drop('category', axis=1).values
        self.tmp_dir = tempfile.mkdtemp()

    def test_prediction(self):
        typer = CellTyper(cache_dir=self.tmp_dir)
        predictions = typer.predict(self.data)
        self.assertEqual(
            predictions.shape,
            (self.data.shape[0], len(CellTyper.celltype_names))
        )

        CellTyper.logits_to_celltype(predictions)
        self.assertEqual(
            predictions.shape,
            (self.data.shape[0], len(CellTyper.celltype_names))
        )

    def tearDown(self):
        """Tear down the tests."""
        shutil.rmtree(self.tmp_dir)


if __name__ == "__main__":
    unittest.main()
