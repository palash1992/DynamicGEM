import theano


def choose(a, choices, out=None, mode='raise'):
    return theano.tensor.choose(a, choices, out, mode)


def tensordot(a, b, axes=2):
    return theano.tensor.tensordot(a, b, axes)


def arange(start, stop=None, step=1, dtype=None):
    return theano.tensor.arange(start, stop, step, dtype)


def logsoftmax(c):
    return theano.tensor.nnet.logsoftmax(c)


def var(inpt, axis=None, keepdims=False):
    return theano.tensor.var(inpt, axis, keepdims)


def eye(n, m=None, k=0, dtype=None):
    return theano.tensor.eye(n, m, k, dtype)


def gamma(a):
    return theano.tensor.gamma(a)


def pool2d_raw(inpt, ds, ignore_border=False, st=None, padding=(0, 0), mode='max'):
    return theano.tensor.signal.pool.pool_2d(inpt, ds, ignore_border, st, padding, mode)
