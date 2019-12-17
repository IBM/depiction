"""Wrapper for pretrained Deepbind via Kipoi."""
import numpy as np
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.language import Language
from concise.preprocessing.sequence import encodeDNA, encodeRNA

from ....core import Task, DataType
from ...kipoi.core import KipoiModel

DEEPBIND_CLASSES = ['NotBinding', 'Binding']
ALPHABET = {'TF': ['T', 'C', 'G', 'A', 'N'], 'RBP': ['U', 'C', 'G', 'A', 'N']}
ONE_HOT_ENCODER = {'TF': encodeDNA, 'RBP': encodeRNA}


def create_sequence_language(alphabet):
    """Anchor accepts a spacy language for sampling the neighborhood."""
    vocab = Vocab(strings=alphabet)

    def make_doc(sequence):
        sequence = sequence.replace(' ', '')
        if len(sequence) == 0:
            words = np.random.choice(alphabet)
        else:
            words = list(sequence)
        return Doc(vocab, words=words, spaces=[False] * len(words))

    return Language(vocab, make_doc)


def create_DNA_language():
    return create_sequence_language(alphabet=ALPHABET['TF'])


def create_RNA_language():
    return create_sequence_language(alphabet=ALPHABET['RBF'])


def character_correction(sequences_list, min_length, null_character='N'):
    """
    Some perturbation based interpretability methods (e.g. lime)
    might introduce null characters which are not viable input.
    These are by default replaced with 'N' (for any character).

    The sequence is padded to min_length characters.
    """
    return [
        s.replace('\x00', null_character).ljust(min_length, null_character)
        for s in sequences_list
    ]


def preprocessing_function(
    nucleotide_sequence, sequence_type, min_length=35, null_character='N'
):
    """One-hot-encode the sequence and allow passing single string."""

    if isinstance(nucleotide_sequence, str):
        sequences_list = [nucleotide_sequence]
    else:
        if not hasattr(nucleotide_sequence, '__iter__'):
            raise IOError(
                f'Expected a str or iterable, got {type(nucleotide_sequence)}.'
            )
        sequences_list = nucleotide_sequence

    return ONE_HOT_ENCODER[sequence_type](
        character_correction(sequences_list, min_length, null_character)
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocessing_function(binding_score, use_labels=True):
    """Instead of a score, interpreters expect labels or probabilities."""
    if use_labels:
        return binding_score > 0  # binding_probs > 0.5
    else:
        # not a score, but probability in [0,1]
        binding_probs = np.expand_dims(sigmoid(binding_score), axis=1)
        return np.hstack([1. - binding_probs, binding_probs])


class DeepBind(KipoiModel):
    """Deepbind wrapper via kipoi."""

    def __init__(self, model, use_labels=True, min_length=0):
        """
        Constructor.

        Args:
            model (string): kipoi model name.
            use_labels (bool): if False, use probabilites instead of label.
            min_length (int): minimal lenght of sequence used for eventual
                padding with null_character ('N'). Some deepbind models fail
                with too short sequences, in that case increase min_length.

        On top of the kipoi model prediction, the predict method of this class
        will preprocess a string sequence to one hot encoding using the
        the input documentation to determine `sequence_type`.
        It will also return not a binding score but either a classification
        label or 'NotBinding','Binding' probabilities expected by interpreters.
        """
        super().__init__(
            model=model,
            task=Task.CLASSIFICATION,
            data_type=DataType.TEXT,
            source='kipoi',
            with_dataloader=False,
            preprocessing_function=preprocessing_function,
            preprocessing_kwargs={},
            postprocessing_function=postprocessing_function,
            postprocessing_kwargs={},
        )
        # kwargs
        self.use_labels = use_labels
        # self.model.schema.inputs.doc is always "DNA Sequence", use name
        self.sequence_type = model.split('/')[2]  # 'TF' or 'RBP'
        self.min_length = min_length

    def predict(self, sample):
        self.preprocessing_kwargs['sequence_type'] = self.sequence_type
        self.preprocessing_kwargs['min_length'] = self.min_length
        self.postprocessing_kwargs['use_labels'] = self.use_labels
        return super().predict(sample)
