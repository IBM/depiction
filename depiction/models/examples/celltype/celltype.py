"""CellTyper model."""
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from ...uri.cache.http_model import HTTPModel
from ....core import Task, DataType


def one_hot_encoding(classes):
    return to_categorical(classes)[:, 1:]  # remove category 0


def one_hot_decoding(labels):
    return labels.argmax(axis=1) + 1


class CellTyper(HTTPModel):
    """Classifier of single cells."""
    celltype_names = {
        1: 'CD11b- Monocyte',
        2: 'CD11bhi Monocyte',
        3: 'CD11bmid Monocyte',
        4: 'Erythroblast',
        5: 'HSC',
        6: 'Immature B',
        7: 'Mature CD38lo B',
        8: 'Mature CD38mid B',
        9: 'Mature CD4+ T',
        10: 'Mature CD8+ T',
        11: 'Megakaryocyte',
        12: 'Myelocyte',
        13: 'NK',
        14: 'Naive CD4+ T',
        15: 'Naive CD8+ T',
        16: 'Plasma cell',
        17: 'Plasmacytoid DC',
        18: 'Platelet',
        19: 'Pre-B II',
        20: 'Pre-B I'
    }

    def __init__(
        self,
        filename='celltype_model.h5',
        origin='https://ibm.box.com/shared/static/5uhttlduaund89tpti4y0ptipr2dcj0h.h5',  # noqa
        cache_dir=None
    ):
        """Initialize the CellTyper."""
        super().__init__(
            uri=origin,
            task=Task.CLASSIFICATION,
            data_type=DataType.TABULAR,
            cache_dir=cache_dir,
            filename=filename
        )
        self.model = keras.models.load_model(self.model_path)

    def predict(self, sample, *args, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        parameters.

        Args:
            sample (object): an input sample for the model.
            args (list): list of arguments.
            kwargs (dict): list of key-value arguments.

        Returns:
            a prediction for the model on the given sample.
        """
        return self.model.predict(
            sample, batch_size=None, verbose=0, steps=None, callbacks=None
        )

    @staticmethod
    def logits_to_celltype(predictions):
        return [
            CellTyper.celltype_names[category]
            for category in one_hot_decoding(predictions)
        ]
