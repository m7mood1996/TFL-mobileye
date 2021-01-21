import tensorflow as tf

import numpy as np


def get_tfl_detaction_model():
    # load the model architecture:
    loaded_model = tf.keras.models.load_model('/Users/mahmoodnael/PycharmProjects/Mobileye/TFL_validate/model.h5')
    # load the weights:
    return loaded_model


def validate_tfl(model, image):
    image = image.reshape(1, 81, 81, 3)

    result = model.predict(image)

    return result[0][0] > 0.85
