"""SMILES utilities."""
import os
import re
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from operator import itemgetter
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

ATOM_REGEX = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|'
    r'-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)
MAXIMUM_NUMBER_OF_RINGS = int(
    os.environ.get('PACCMANN_MAXIMUM_NUMBER_OF_RINGS', 9)
)
MAX_LENGTH = int(os.environ.get('PACCMANN_MAX_LENGTH', 155))
PADDING_ATOM = os.environ.get('PACCMANN_PADDING_ATOM', '<PAD>')
NON_ATOM_CHARACTERS = set(
    [str(index)
     for index in range(1, MAXIMUM_NUMBER_OF_RINGS)] + ['(', ')', '#', '=']
)
ATOM_MAPPING = {
    '2': 1,
    '7': 2,
    'O': 3,
    '[O]': 4,
    '#': 5,
    '(': 6,
    'P': 7,
    'Cl': 8,
    'C': 9,
    'N': 10,
    'Br': 11,
    'F': 12,
    ')': 13,
    '=': 14,
    '9': 15,
    '4': 16,
    '1': 17,
    '6': 18,
    'I': 19,
    '[N+]': 20,
    '[NH]': 21,
    '.': 22,
    'S': 23,
    '[O-]': 24,
    '3': 25,
    '8': 26,
    '5': 27,
    PADDING_ATOM: 0
}
REVERSED_ATOM_MAPPING = {index: atom for atom, index in ATOM_MAPPING.items()}
CMAP = cm.Oranges
COLOR_NORMALIZERS = {
    'linear': mpl.colors.Normalize,
    'logarithmic': mpl.colors.LogNorm
}
ATOM_RADII = float(os.environ.get('PACCMANN_ATOM_RADII', .5))
SVG_WIDTH = int(os.environ.get('PACCMANN_SVG_WIDTH', 400))
SVG_HEIGHT = int(os.environ.get('PACCMANN_SVG_HEIGHT', 200))
COLOR_NORMALIZATION = os.environ.get(
    'PACCMANN_COLOR_NORMALIZATION', 'logarithmic'
)


def process_smiles(smiles):
    """
    Process a SMILES.

    SMILES string is processed to generate a zero-padded
    sequence.

    Args:
        smiles (str): a SMILES representing a molecule.
    Returns:
        a list of token indices.append()
    """
    tokens = [token for token in ATOM_REGEX.split(smiles)
              if token][:MAX_LENGTH]
    return (
        [0] * (MAX_LENGTH - len(tokens)) +
        [ATOM_MAPPING.get(token, 0) for token in tokens]
    )


def get_atoms(smiles):
    """
    Process a SMILES.

    SMILES string is processed to generate a sequence
    of atoms.

    Args:
        smiles (str): a SMILES representing a molecule.
    Returns:
        a list of atoms.
    """
    tokens = process_smiles(smiles)
    return [REVERSED_ATOM_MAPPING[token] for token in tokens]


def remove_padding_from_atoms_and_smiles_attention(atoms, smiles_attention):
    """
    Remove padding atoms and corresponding attention weights.
    
    Args:
        atoms (Iterable): an iterable of atoms.
        smiles_attention (Iterable): an iterable of floating point values.
    Returns:
        two iterables of atoms and attention values removing the padding.
    """
    to_keep = [
        index for index, atom in enumerate(atoms) if atom != PADDING_ATOM
    ]
    return (
        list(itemgetter(*to_keep)(atoms)),
        list(itemgetter(*to_keep)(smiles_attention))
    )


def _get_index_and_colors(values, objects, predicate, color_mapper):
    """
    Get index and RGB colors from a color map using a rule.

    The predicate acts on a tuple of (value, object).

    Args:
        values (Iterable): floats representing a color.
        objects (Iterable): objects associated to the colors.
        predicate (Callable): a predicate to filter objects.
        color_mapper (cm.ScalarMappable): a mapper from floats to RGBA.

    Returns:
        Iterables of indices and RGBA colors.
    """
    indices = []
    colors = {}
    for index, value in enumerate(
        map(
            lambda t: t[0],
            filter(lambda t: predicate(t), zip(values, objects))
        )
    ):
        indices.append(index)
        colors[index] = color_mapper.to_rgba(value)
    return indices, colors


def smiles_attention_to_svg(smiles_attention, atoms, molecule):
    """
    Generate an svg of the molecule highlighiting SMILES attention.

    Args:
        smiles_attention (Iterable): an iterable of floating point values.
        atoms (Iterable): an iterable of atoms.
        molecule (rdkit.Chem.Mol): a molecule.
    Returns:
        the svg of the molecule with highlighted atoms and bonds.
    """
    # remove padding
    logger.debug('SMILES attention:\n{}'.format(smiles_attention))
    logger.debug(
        'SMILES attention range: [{},{}]'.format(
            min(smiles_attention), max(smiles_attention)
        )
    )
    atoms, smiles_attention = remove_padding_from_atoms_and_smiles_attention(
        atoms, smiles_attention
    )
    logger.debug(
        'atoms and SMILES after removal:\n{}\n{}'.format(
            atoms, smiles_attention
        )
    )
    logger.debug(
        'SMILES attention after padding removal:\n{}'.format(smiles_attention)
    )
    logger.debug(
        'SMILES attention range after padding removal: [{},{}]'.format(
            min(smiles_attention), max(smiles_attention)
        )
    )
    # define a color map
    normalize = COLOR_NORMALIZERS.get(COLOR_NORMALIZATION, mpl.colors.LogNorm)(
        vmin=min(smiles_attention), vmax=2 * max(smiles_attention)
    )
    color_mapper = cm.ScalarMappable(norm=normalize, cmap=CMAP)
    # get atom colors
    highlight_atoms, highlight_atom_colors = _get_index_and_colors(
        smiles_attention, atoms, lambda t: t[1] not in NON_ATOM_CHARACTERS,
        color_mapper
    )
    logger.debug('Atom colors:\n{}'.format(highlight_atom_colors))
    # get bond colors
    highlight_bonds, highlight_bond_colors = _get_index_and_colors(
        smiles_attention, atoms, lambda t: t[1] in NON_ATOM_CHARACTERS,
        color_mapper
    )
    logger.debug('Bond colors:\n{}'.format(highlight_bond_colors))
    # add coordinates
    Chem.rdDepictor.Compute2DCoords(molecule)
    # draw the molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(SVG_WIDTH, SVG_HEIGHT)
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii={index: ATOM_RADII
                            for index in highlight_atoms}
    )
    drawer.FinishDrawing()
    # return the drawn molecule
    return drawer.GetDrawingText().replace('\n', ' ')


def get_smiles_language():
    """
    Get SMILES language.
    
    Returns:
        a spacy.language.Language representing SMILES.
    """
    valid_values = list(
        filter(lambda k: k != PADDING_ATOM, ATOM_MAPPING.keys())
    )
    vocabulary = Vocab(strings=valid_values)

    def make_doc(smiles):
        """
        Make a SMILES document.

        Args:
            smiles (str): a SMILES representing a molecule.
        Returns:
            a spacy.tokens.Doc representing the molecule.
        """
        if len(smiles) == 0:
            tokens = np.random.choice(valid_values)
        else:
            tokens = [token for token in ATOM_REGEX.split(smiles)
                      if token][:MAX_LENGTH]
        return Doc(vocabulary, words=tokens, spaces=[False] * len(tokens))

    return Language(vocabulary, make_doc)
