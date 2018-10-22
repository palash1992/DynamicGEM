import keras.backend as K

if K._BACKEND == 'tensorflow':
    from tensorflow.python import debug as tfdbg
    sess = tfdbg.LocalCLIDebugWrapperSession(K.get_session())
    K.set_session(sess)
    sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
else:
    raise RuntimeError("finite_number_check not avaiable for backend {}".format(K._BACKEND))
