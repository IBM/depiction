from .core import Model
from ..core import Task
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


def one_hot_encoding(classes):
    return to_categorical(classes)[:, 1:] # remove category 0


def one_hot_decoding(labels):
    return labels.argmax(axis=1) + 1


class CellTyper(Model):
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
        20: 'Pre-B I'}

    def __init__(self, *args, **kwargs):
        """Initalize the Model."""
        super(__class__).__init__(self, Task.CLASSIFICATION, *args, **kwargs)
        # TODO wget model.h5
        self.model = keras.models.load_model('../data/models/celltype_dnn_model.h5')

    def predict(self, sample, **kwargs):
        """
        Run the model for inference on a given sample and with the provided
        arameters.
        
        Arguments:
            sample (object): an input sample for the model.
            kwargs (dict): list of key-value arguments.

        Returns
            a prediction for the model on the given sample.
        """
        y = self.model.predict(
            sample,
            batch_size=None, verbose=0, steps=None, callbacks=None
        )
        return one_hot_decoding(y)

    @staticmethod
    def to_cellnames(predictions):
        return [CellTyper.celltype_names[category] for category in predictions]