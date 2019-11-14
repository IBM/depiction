"""UWahsers for images."""
# %%
import json
import numpy as np
import keras_applications
from tensorflow import keras

from depiction.core import DataType, Task
from depiction.models.keras import KerasApplicationModel
from depiction.interpreters.u_wash import UWasher


# %%
# general utils
def image_preprocessing(image_path, preprocess_input, target_size):
    """
    Read and preprocess an image from disk.

    Args:
        image_path (str): path to the image.
        preprocess_input (funciton): a preprocessing function.
        target_size (tuple): image target size.

    Returns:
        np.ndarray: the preprocessed image.
    """
    image = keras.preprocessing.image.load_img(
        image_path, target_size=target_size
    )
    x = keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def get_imagenet_labels():
    """
    Get ImamgeNet labels.

    Returns:
        list: list of labels.
    """
    labels_filepath = keras.utils.get_file(
        'imagenet_class_index.json',
        keras_applications.imagenet_utils.CLASS_INDEX_PATH
    )
    with open(labels_filepath) as fp:
        labels_json = json.load(fp)
    labels = [None] * len(labels_json)
    for index, (_, label) in labels_json.items():
        labels[int(index)] = label
    return labels


# %%
labels = get_imagenet_labels()

# %%
# instantiate the model
model = KerasApplicationModel(
    keras.applications.MobileNetV2(), Task.CLASSIFICATION, DataType.IMAGE
)

#%%
image_path = keras.utils.get_file(
    'elephant.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Zoorashia_elephant.jpg/120px-Zoorashia_elephant.jpg'  # noqa
)
image = image_preprocessing(
    image_path,
    keras.applications.mobilenet_v2.preprocess_input,
    target_size=(224, 224)
)

# LIME
# %%
interpreter = UWasher('lime', model, class_names=labels)

# %%
explanation = interpreter.interpret(image.squeeze())

# Anchors
# %%
interpreter = UWasher('anchors', model)

# %%
explanation = interpreter.interpret(image.squeeze())
