"""Wrapping PaccMann model."""
import os
import json
import zipfile
import numpy as np
import tensorflow as tf
from copy import deepcopy
from paccmann.models import paccmann_model_fn, MODEL_SPECIFICATION_FACTORY

from .smiles import process_smiles, get_smiles_language
from ....core import Task, DataType
from ...uri.cache.http_model import HTTPModel

MODEL_PARAMS_JSON = 'model_params.json'
MODEL_CHECKPOINT = 'model.ckpt-375000'
NUMBER_OF_GENES = 2128
SMILES_LENGTH = 155


class CachedGraphPaccMannPredictor(object):
    """
    A paccmann predictor that caches the graph avoiding reloading it.

    Inspired by:
    https://github.com/marcsto/rl/blob/master/src/fast_predict2.py
    """

    def __init__(self, estimator, input_fn, batch_size, checkpoint_path=None):
        """
        Initialize CachedGraphPaccMannPredictor.

        Args:
            estimator (tf.estimator.Estimator): a PaccMann estimator.
            input_fn (Callable): an input function accepting a generator.
            batch_size (int): size of the batch supported by the estimator.
            checkpoint_path (str): path to the checkpoint to load.
                Defaults to loading the latest.
        """
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

    def _create_generator(self):
        """
        Create a generator of examples.

        Returns:
            a generator of example processed in self.input_fn.
        """
        while not self.closed:
            for example in self.next_examples:
                yield example

    def predict(self, examples):
        """
        Predict on the given examples.

        Args:
            examples (Iterable): an iterable of examples.

        Returns:
            a np.array representing the logits [non-effective, effective].
        """
        # make sure we don't alter the input
        examples = deepcopy(examples)
        if isinstance(examples, np.ndarray) and len(examples.shape) > 1:
            examples = [row for row in examples]
        if not isinstance(examples, list):
            examples = [examples]
        number_of_examples = len(examples)
        iterations = number_of_examples // self.batch_size
        remainder = number_of_examples % self.batch_size
        # handle remainder in samples
        if remainder > 0:
            examples += [examples[-1]] * (self.batch_size - remainder)
        self.next_examples = examples
        if self.first_run:
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator()),
                yield_single_examples=False,
                checkpoint_path=self.checkpoint_path
            )
            self.first_run = False
        results = []
        for _ in range(iterations):
            prediction = next(self.predictions)
            results.extend([value, 1 - value] for value in prediction['IC50'])
        # get rid of the optional remainder
        if remainder > 0:
            prediction = next(self.predictions)
            results.extend(
                [value, 1 - value] for value in prediction['IC50'][:remainder]
            )
        return np.array(results)

    def close(self):
        """
        Close the predictor.
        """
        self.closed = True
        try:
            next(self.predictions)
        except Exception:
            pass


def generator_fn_to_dataset(
    generator_fn, number_of_genes, smiles_length, batch_size
):
    """
    Get a tf.data.Dataset from a generator function.

    Args:
        generator_fn (Callable): a generator function.
        number_of_genes (int): number of the gene considered.
        smiles_length (int): maximum SMILES length.
        batch_size (int): size of the batch.
    Returns:
        An input function.
    """
    return tf.data.Dataset.from_generator(
        generator_fn,
        output_types={
            'selected_genes_20': tf.float32,
            'smiles_atom_tokens': tf.int64
        },
        output_shapes={
            'selected_genes_20': number_of_genes,
            'smiles_atom_tokens': smiles_length
        }
    ).batch(batch_size)


def paccmann_smiles_input_fn(
    generator, cell_line, number_of_genes, smiles_length, batch_size
):
    """
    Input for SMILES data.

    Args:
        generator (generator): a generator providing SMILES.
        cell_line (np.ndarray): an array containing cell line gene expression.
        number_of_genes (int): number of the gene considered.
        smiles_length (int): maximum SMILES length.
        batch_size (int): size of the batch.
    Returns:
        an input function accepting a SMILES generator.
    """

    def generator_fn():
        for smiles in generator:
            yield {
                'selected_genes_20': cell_line.astype(np.float32),
                'smiles_atom_tokens': np.array(process_smiles(smiles))
            }

    return lambda: generator_fn_to_dataset(
        generator_fn, number_of_genes, smiles_length, batch_size
    )


def paccmann_cell_line_input_fn(
    generator, smiles, number_of_genes, smiles_length, batch_size
):
    """
    Input for cell line data.

    Args:
        generator (generator): a generator providing cell line data.
        smiles (str): a SMILES representing a molecule.
        number_of_genes (int): number of the gene considered.
        smiles_length (int): maximum SMILES length.
        batch_size (int): size of the batch.
    Returns:
        an input function accepting a cell line generator.
    """

    def generator_fn():
        for cell_line in generator:
            yield {
                'selected_genes_20': cell_line.astype(np.float32),
                'smiles_atom_tokens': np.array(process_smiles(smiles))
            }

    return lambda: generator_fn_to_dataset(
        generator_fn, number_of_genes, smiles_length, batch_size
    )


class PaccMann(HTTPModel):
    """Multimodal classification of drug sensitivity."""

    def __init__(
        self,
        data_type,
        filename='paccmann.zip',
        origin='https://ibm.box.com/shared/static/dy2x4cen1dsrc738uewmv1iccdawlqwd.zip',
        model_type='mca',
        model_params_json='model_params.json',
        model_checkpoint='model.ckpt-375000',
        number_of_genes=2128,
        smiles_length=155,
        cache_dir=None
    ):
        """
        Initialize PaccMann.

        Args:
            data_type (depiction.core.DataType): data type.
            filename (str): model zip.
            origin (str): url where the model is stored.
            model_type (str): multimodal encoder type.
            model_params_json (str): name of the json containing the
                parameters.
            model_checkpoints (str): name of the checkpoint.
            number_of_genes (int): number of the gene considered.
            smiles_length (int): maximum SMILES length.
            args (Iterable): list of arguments.
            cache_dir (str): cache directory.
        """
        super().__init__(
            uri=origin,
            task=Task.CLASSIFICATION,
            data_type=data_type,
            cache_dir=cache_dir,
            filename=filename
        )
        # store initalization parameters
        self.model_type = model_type
        self.model_params_json = model_params_json
        self.model_checkpoint = model_checkpoint
        self.number_of_genes = number_of_genes
        self.smiles_length = smiles_length
        # make sure the model is present
        self.model_dir = os.path.join(
            os.path.dirname(self.model_path), 'paccmann'
        )
        if not os.path.exists(self.model_dir):
            with zipfile.ZipFile(self.model_path) as zip_fp:
                zip_fp.extractall(os.path.dirname(self.model_path))
        # parse the parameters
        self.params = None
        with open(os.path.join(self.model_dir, self.model_params_json)) as fp:
            self.params = json.load(fp)
        self.batch_size = self.params['eval_batch_size']
        self.checkpoint_path = os.path.join(
            self.model_dir, self.model_checkpoint
        )
        # initialize the estimator
        self.estimator = tf.estimator.Estimator(
            model_fn=(
                lambda features, labels, mode, params: paccmann_model_fn(
                    features,
                    labels,
                    mode,
                    params,
                    model_specification_fn=(  # yapf-disable
                        MODEL_SPECIFICATION_FACTORY[self.model_type]
                    )
                )
            ),
            model_dir=self.model_dir,
            params=self.params
        )
        # create the SMILES language
        self.language = get_smiles_language()
        # placeholder for a predictor
        self.predictor = None

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
        return self.predictor.predict(sample)


class PaccMannSmiles(PaccMann):
    """Multimodal classification of drug sensitivity for a given cell line."""

    def __init__(
        self,
        cell_line,
        filename='paccmann.zip',
        origin='https://ibm.box.com/shared/static/dy2x4cen1dsrc738uewmv1iccdawlqwd.zip',
        model_type='mca',
        model_params_json='model_params.json',
        model_checkpoint='model.ckpt-375000',
        number_of_genes=2128,
        smiles_length=155,
        cache_dir=None
    ):
        """
        Initialize the Model.

        Args:
            cell_line (np.ndarray): an array containing cell line gene
                expression.
            filename (str): model zip.
            origin (str): url where the model is stored.
            model_type (str): multimodal encoder type.
            model_params_json (str): name of the json containing the
                parameters.
            model_checkpoints (str): name of the checkpoint.
            number_of_genes (int): number of the gene considered.
            smiles_length (int): maximum SMILES length.
            cache_dir (str): cache directory.
        """
        self.cell_line = cell_line
        super().__init__(
            DataType.TEXT, filename, origin, model_type, model_params_json,
            model_checkpoint, number_of_genes, smiles_length, cache_dir
        )
        self.input_fn = lambda generator: paccmann_smiles_input_fn(
            generator, self.cell_line, self.number_of_genes, self.
            smiles_length, self.batch_size
        )
        self.predictor = CachedGraphPaccMannPredictor(
            self.estimator, self.input_fn, self.batch_size,
            self.checkpoint_path
        )


class PaccMannCellLine(PaccMann):
    """Multimodal classification of drug sensitivity for a given SMILES."""

    def __init__(
        self,
        smiles,
        filename='paccmann.zip',
        origin='https://ibm.box.com/shared/static/dy2x4cen1dsrc738uewmv1iccdawlqwd.zip',
        model_type='mca',
        model_params_json='model_params.json',
        model_checkpoint='model.ckpt-375000',
        number_of_genes=2128,
        smiles_length=155,
        cache_dir=None
    ):
        """
        Initialize the Model.

        Args:
            smiles (str): a SMILES representing a molecule.
            filename (str): model zip.
            origin (str): url where the model is stored.
            model_type (str): multimodal encoder type.
            model_params_json (str): name of the json containing the
                parameters.
            model_checkpoints (str): name of the checkpoint.
            number_of_genes (int): number of the gene considered.
            smiles_length (int): maximum SMILES length.
            cache_dir (str): cache directory.
        """
        self.smiles = smiles
        super().__init__(
            DataType.TABULAR, filename, origin, model_type, model_params_json,
            model_checkpoint, number_of_genes, smiles_length, cache_dir
        )
        self.input_fn = lambda generator: paccmann_cell_line_input_fn(
            generator, self.smiles, self.number_of_genes, self.smiles_length,
            self.batch_size
        )
        self.predictor = CachedGraphPaccMannPredictor(
            self.estimator, self.input_fn, self.batch_size,
            self.checkpoint_path
        )
