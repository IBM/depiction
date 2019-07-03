import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from paccmann.models import paccmann_model_fn, MODEL_SPECIFICATION_FACTORY
from paccmann.learning import eval_input_fn
from depiction.models.paccmann.smiles import process_smiles


#%% input data
# drugs
drugs = pd.read_csv('data/paccmann/gdsc.smi', sep='\t', index_col=1)
# cell lines
cell_lines = pd.read_csv('data/paccmann/gdsc.csv.gz', index_col=0)
#%%
# path to the model
model_path = '/scratch/paccmann/ismb-eccb_tutorial/paccmann'
MODEL_PARAMS_JSON = 'model_params.json'
MODEL_CHECKPOINT = 'model.ckpt-375000'
# read model parameters
params = None
with open(os.path.join(model_path, 'model_params.json')) as fp:
    params = json.load(fp)
# get constants
BATCH_SIZE = params['eval_batch_size']
SMILES = drugs.iloc[0].item()
CELL_LINE = cell_lines.iloc[0][3:].values
# initialize the estimator
estimator = tf.estimator.Estimator(
    model_fn=(
        lambda features, labels, mode, params: paccmann_model_fn(
            features, labels, mode, params,
            model_specification_fn=(
                MODEL_SPECIFICATION_FACTORY['mca']
            )
        )
    ),
    model_dir=model_path,
    params=params
)
#%% get a pair
smiles = drugs.iloc[0].item()
cell_line = cell_lines.iloc[0][3:].values

#%% # setup predictions
class Predictor(object):
    """Inspired by https://github.com/marcsto/rl/blob/master/src/fast_predict2.py."""

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            for example in self.next_examples:
                yield example

    def predict(self, examples):
        self.next_examples = examples
        number_of_examples = len(examples)
        iterations = number_of_examples // BATCH_SIZE
        remainder = number_of_examples % BATCH_SIZE
        # handle remainder in samples
        if remainder > 0:
            examples += [examples[-1]]*remainder
        if self.first_run:
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator()),
                yield_single_examples=False,
                checkpoint_path=os.path.join(model_path, MODEL_CHECKPOINT)
            )
            self.first_run = False
        results = []
        for _ in range(iterations):
            prediction = next(self.predictions)
            results.extend(
                [value, 1 - value]
                for value in prediction['IC50']
            )
        # get rid of the optional remainder
        if remainder > 0:
            prediction = next(self.predictions)
            results.extend(
                [value, 1 - value]
                for value in prediction['IC50'][:remainder]
            )
        return np.array(results)

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            pass


def paccmann_smiles_input_fn(generator):
    """ 
    Input for smiles data.
    
    Arguments:
        generator (generator): a generator providing smiles.

    Returns:
        the input function.
    """
    def generator_fn():
        for smiles in generator:
            yield {
                'selected_genes_20': CELL_LINE.astype(np.float32),
                'smiles_atom_tokens': np.array(process_smiles(smiles))
            }

    def _input_fn():
        return tf.data.Dataset.from_generator(
            generator_fn,
            output_types={
                'selected_genes_20': tf.float32,
                'smiles_atom_tokens': tf.int64
            },
            output_shapes={
                'selected_genes_20': 2128,
                'smiles_atom_tokens': 155
            }
        ).batch(BATCH_SIZE)

    return _input_fn


def paccmann_cell_line_input_fn(generator):
    """ 
    Input for cell line data.
    
    Arguments:
        generator (generator): a generator providing cell line data.

    Returns:
        the input function.
    """
    def generator_fn():
        for cell_line in generator:
            yield {
                'selected_genes_20': cell_line.astype(np.float32),
                'smiles_atom_tokens': np.array(process_smiles(SMILES))
            }

    def _input_fn():
        return tf.data.Dataset.from_generator(
            generator_fn,
            output_types={
                'selected_genes_20': tf.float32,
                'smiles_atom_tokens': tf.int64
            },
            output_shapes={
                'selected_genes_20': 2128,
                'smiles_atom_tokens': 155
            }
        ).batch(BATCH_SIZE)

    return _input_fn


#%%
predictor = Predictor(estimator, paccmann_smiles_input_fn)
some_drugs = list(drugs.values[:,0][:100])
#%%
results = predictor.predict(some_drugs)
