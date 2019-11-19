# %% [markdown]
# # Contrastive Explanations Method (CEM) applied to MNIST

# %% [markdown]
# The Contrastive Explanation Method (CEM) can generate black box model explanations in terms of pertinent positives (PP) and pertinent negatives (PN). For PP, it finds what should be minimally and sufficiently present (e.g. important pixels in an image) to justify its classification. PN on the other hand identify what should be minimally and necessarily absent from the explained instance in order to maintain the original prediction.
#
# The original paper where the algorithm is based on can be found on [arXiv](https://arxiv.org/pdf/1802.07623.pdf).
# Depiction wraps an implementation by the alibi package and follows their [example](https://docs.seldon.io/projects/alibi/en/stable/examples/cem_mnist.html) heavily.
# %%

import tempfile
import random
# import pandas as pd
import numpy as np

import matplotlib
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
# import ipywidgets as widgets
# from ipywidgets import interact, interact_manual
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # suppress deprecation messages
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from depiction.models.base.utils import get_model_file
from depiction.models.uri.cache.http_model import HTTPModel
from depiction.core import Task, DataType
from depiction.interpreters.alibi.contrastive.cem import CEM

# %% [markdown]
# ## Load MNIST data

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
plt.gray()
# plt.imshow(x_test[4]);

# %% [markdown]
# Models were trained as shown in [alibis example](https://docs.seldon.io/projects/alibi/en/stable/examples/cem_mnist.html)
# which we follow but using depiction with the benefit of using its other interpreters
# %%


class MNISTClassifier(HTTPModel):

    def __init__(
        self,
        filename='mninst_cnn.h5',
        origin='https://ibm.box.com/shared/static/k1x70cmr01fahob5ub7y2r82jqv3r75b.h5',  # noqa
        cache_dir=tempfile.mkdtemp()
    ):
        """Initialize the CellTyper."""
        super().__init__(
            uri=origin,
            task=Task.CLASSIFICATION,
            data_type=DataType.IMAGE,
            cache_dir=cache_dir,
            filename=filename
        )
        self.model = keras.models.load_model(self.model_path)

    def predict(self, sample):
        return self.model.predict(sample)


# %%
# A model from depiction for interpretation
cnn = MNISTClassifier()
cnn.model.summary()
# %%
# CEM accepts an optional keras autoencoder to find better a pertinent
# negative/positive

ae = keras.models.load_model(
    get_model_file(
        filename='mninst_ae.h5',
        origin=
        'https://ibm.box.com/shared/static/psogbwnx1cz0s8w6z2fdswj25yd7icpi.h5',  # noqa
        cache_dir=cnn.cache_dir
    )
)

ae.summary()

# %% [markdown]
# The models were trained and expext processed data
# so we scale, reshape and categorize

# %%


def transform(x):
    """Move to -0.5, 0.5 range and add channel dimension."""
    return np.expand_dims(x.astype('float32') / 255 - 0.5, axis=-1)


def transform_sample(x):
    return np.expand_dims(transform(x), axis=0)


def inverse_transform(data):
    return (data.squeeze() + 0.5) * 255


def show_image(x):
    return plt.imshow(x.squeeze())


# %% [markdown]
# Compare original with decoded images

# %%
score = cnn.model.evaluate(
    transform(x_test), to_categorical(y_test), verbose=0
)
print('Test accuracy: ', score[1])

# %% ------------------------------------------

decoded_imgs = ae.predict(transform(x_test))
n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(2, n, i)
    # show_image(transform(x_test[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    # show_image(transform(decoded_imgs[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# plt.show()

# %% [markdown]
# ## Generate contrastive explanation with pertinent negative
# %% [markdown]
# Explained instance:

# %%
idx = 15
X = transform_sample(x_test[idx])

# %%
# show_image(X)

# %% [markdown]
# Model prediction:

# %%
cnn.predict(X).argmax(), cnn.predict(X).max()

# %% [markdown]
# CEM parameters:

# %%
mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
shape = X.shape  # instance shape, batchsize must be 1
assert shape[0] == 1
kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
# class predicted by the original instance and the max probability on the other classes
# in order for the first loss term to be minimized
beta = .1  # weight of the L1 loss term
gamma = 100  # weight of the optional auto-encoder loss term
c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or
# the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = 10  # nb of updates for c
max_iterations = 1000  # nb of iterations per value of c
feature_range = (
    x_train.min(), x_train.max()
)  # feature range for the perturbed instance
clip = (-1000., 1000.)  # gradient clipping
lr = 1e-2  # initial learning rate
no_info_val = -1.  # a value, float or feature-wise, which can be seen as containing no info to make a prediction
# perturbations towards this value means removing features, and away means adding features
# for our MNIST images, the background (-0.5) is the least informative,
# so positive/negative perturbations imply adding/removing features

# %% [markdown]
# Generate pertinent negative:

# %%
# initialize CEM explainer and explain instance
cem = CEM(
    cnn,
    mode,
    shape,
    kappa=kappa,
    beta=beta,
    feature_range=feature_range,
    gamma=gamma,
    ae_model=ae,
    max_iterations=max_iterations,
    c_init=c_init,
    c_steps=c_steps,
    learning_rate_init=lr,
    clip=clip,
    no_info_val=no_info_val
)

explanation = cem.interpret(X, verbose=True)

# %% [markdown]
# Pertinent negative:

# %%
print('Pertinent negative prediction: {}'.format(explanation[mode + '_pred']))
# show_image(explanation[mode]);

# %% [markdown]
# ## Generate pertinent positive

# %%
mode = 'PP'

# %%
# initialize CEM explainer and explain instance
cem = CEM(
    cnn,
    mode,
    shape,
    kappa=kappa,
    beta=beta,
    feature_range=feature_range,
    gamma=gamma,
    ae_model=ae,
    max_iterations=max_iterations,
    c_init=c_init,
    c_steps=c_steps,
    learning_rate_init=lr,
    clip=clip,
    no_info_val=no_info_val
)

explanation = cem.interpret(X, verbose=True)

# %% [markdown]
# Pertinent positive:

# %%
print('Pertinent positive prediction: {}'.format(explanation[mode + '_pred']))
# show_image(explanation[mode]);

# %%
# to delete the downloaded files before your next reboot
# import os
# os.remove(cnn.model_path)
# os.remove(os.path.join(cnn.cache_dir, 'mninst_ae.h5'))
