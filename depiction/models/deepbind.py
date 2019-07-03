"""
Wrapper for pretrained Deepbind
"""

import os
import math
import tempfile
import subprocess
import numpy as np
from core import Model
from subprocess import PIPE
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ..core import Task, DataType
from spacy.language import Language


DEEPBIND_ENV = "DEEPBIND_LOCATION"
DEEPBIND_CLASSES = ['NotBinding', 'Binding']
SEQ_FILE_EXTENSION = ".seq"
DNA_ALPHABET = ['T', 'C', 'G', 'A', 'U', 'N']


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def process_deepbind_stdout(deepbind_stdout):
    """
    Process the output assuming that there is only one input and one factor, i.e.
    the output has this format:
    
    <BOF>
    <FACTOR_ID>\n
    <binding_score>\n
    <EOF>

    Returns:
    Probability of binding, as sigmoid(binding score) 
    """
    return np.expand_dims(sigmoid(np.array(deepbind_stdout.splitlines()[1:]).astype(np.float)), axis=1)


def deepbind(factor_id, sequence_fpath):
    exec_path = os.getenv(DEEPBIND_ENV)
    process = subprocess.run([exec_path, factor_id, sequence_fpath], stdout=PIPE, stderr=PIPE)

    return process_deepbind_stdout(process.stdout)


def deepbind_on_sequences(factor_id, sequences_list, tmp_folder = None):
    tmp_file = tempfile.mkstemp(dir=tmp_folder, suffix = SEQ_FILE_EXTENSION)[1]

    with open(tmp_file, 'w') as tmp_fh:
        tmp_fh.write('\n'.join([s.replace('\x00', '') if len(s.replace('\x00', '')) > 0 else np.random.choice(DNA_ALPHABET) for s in sequences_list]))

    return deepbind(factor_id, tmp_file)


def create_DNA_language():
    accepted_values = DNA_ALPHABET
    vocab = Vocab(strings=accepted_values)

    def make_doc(sequence):
        sequence = sequence.replace(' ', '')
        if len(sequence) == 0:
            words = np.random.choice(accepted_values)
        else:
            words = list(sequence)
        return Doc(vocab, words=words, spaces=[False]*len(words))

    return Language(vocab, make_doc)


class DeepBind(Model):
    """
    Deepbind wrapper
    """
    def __init__(self, tf_factor_id, use_labels, tmp_folder = None):
        """
        Constructor

        Args:
            tf_factor_id (str): ID of the transcription factor to classify against.
            use_labels (bool): if False, use logits insted of labels.
            tmp_folder (str): where to store temporary files created. Defaults to the system default.
        """
        super(DeepBind).__init__(self, Task.CLASSIFICATION, DataType.TEXT)
        self.tf_factor_id = tf_factor_id
        self.tmp_folder = tmp_folder
        self.use_labels = use_labels

    def predict(self, sample):
        if not isinstance(sample, list):
            sample = [sample]            
        binding_probs = deepbind_on_sequences(self.tf_factor_id, sample)
        if use_labels:
            return binding_probs.flatten() > 0.5
        else:
            return np.hstack([1. - binding_probs,  binding_probs])