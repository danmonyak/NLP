"""
wrappers.py
=======
Author - Daniel Monyak
=======

Module containing wrapper functions
"""

import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)


