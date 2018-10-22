from __future__ import absolute_import

import warnings

import keras.backend

if keras.backend._BACKEND == 'theano':
    from . import theano_patches
    for k, v in vars(theano_patches).items():
        if callable(v):
            setattr(keras.backend, k, v)
elif keras.backend._BACKEND == 'tensorflow':
    from . import tensorflow_patches
    for k, v in vars(tensorflow_patches).items():
        if callable(v):
            setattr(keras.backend, k, v)
else:
    warnings.warn("Unknown backend type: {}".format(keras.backend._BACKEND))
