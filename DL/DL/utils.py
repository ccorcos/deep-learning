#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
import time

def load_data(dataset, types):
    ''' Loads the dataset to the GPU

    dataset = [train_set, valid_set, test_set]
    
    each set is a tuple (input, target)
    input is a matrix where rows are a sample
    target is a 1d array of what output should be

    types is an array of types "int32" or "float32"
    '''

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        Create a shared dataset, copying the whole thing to the GPU.
        We dont want to copy each minibatch over one at a time.
        """
        sharedData = []
        for data, t, in zip(data_xy, types):
            shared = theano.shared(numpy.asarray(data,
                                                   dtype=theano.config.floatX),
                                                   borrow=borrow)
            sharedData.append(shared)

        # You have to store values on the GPU as floats. But the y is 
        # really an int so we'll cast back to an int for what we return
        if t == 'int32':
            sharedData[-1] = T.cast(sharedData[-1], 'int32')
        
        return sharedData

    test_set = shared_dataset(dataset[2])
    valid_set = shared_dataset(dataset[1])
    train_set = shared_dataset(dataset[0])

    rval = [train_set, valid_set, test_set]
    return rval

def maybe(func, otherwise=None):
    res = None
    try:
        res = func()
    except:
        return otherwise
    return res

def flattenIterator(container):
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten(i):
                yield j
        else:
            yield i

flatten = lambda x: list(flattenIterator(x))

fmt = lambda x: "{:12.8f}".format(x)

def onehot(value, length):
    v = [0]*length
    v[value] = 1
    return v

relu = lambda x: T.switch(x<0, 0, x)
cappedrelu =  lambda x: T.minimum(T.switch(x<0, 0, x), 6)
sigmoid = T.nnet.sigmoid
tanh = T.tanh
# softmax = T.nnet.softmax

# a differentiable version for HF that doesn't have some optimizations.
def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True)) 
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out

activations = {
    'relu': relu,
    'cappedrelu': cappedrelu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'linear': lambda x: x,
    'softmax': softmax
}

def compute_L1(weights):
    return reduce(operator.add, map(lambda x: abs(x).sum(), weights), 0)

def compute_L2_sqr(weights):
    return reduce(operator.add, map(lambda x: (x ** 2).sum(), weights), 0)

def layers_L1(layers):
    return reduce(operator.add, map(lambda x: x.L1, layers), 0)

def layers_L2_sqr(layers):
    return reduce(operator.add, map(lambda x: x.L2_sqr, layers), 0)

def layers_params(layers):
    return reduce(operator.add, map(lambda x: x.params, layers), [])

def mse(output, targets):
    return T.mean((output - targets) ** 2)

def nll_binary(output, targets):
    # negative log likelihood based on binary cross entropy error
    return T.mean(T.nnet.binary_crossentropy(output, targets))    

def nll_multiclass(output, targets):
    return -T.mean(T.log(output)[T.arange(targets.shape[0]), targets])

def nll_multiclass_timeseries(output, targets):    
    # Theano's advanced indexing is limited
    # therefore we reshape our n_steps x n_seq x n_classes tensor3 of probs
    # to a (n_steps * n_seq) x n_classes matrix of probs
    # so that we can use advanced indexing (i.e. get the probs which
    # correspond to the true class)
    # the labels targets also must be flattened when we do this to use the
    # advanced indexing
    p_y = output
    p_y_m = T.reshape(p_y, (p_y.shape[0] * p_y.shape[1], -1))
    y_f = targets.flatten(ndim=1)
    return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])

def pred_binary(output):
    return T.round(output)  # round to {0,1}

def pred_multiclass(output):
    return T.argmax(output, axis=-1)

def pred_error(pred, targets):
    # check if y has same dimension of y_pred
    if targets.ndim != pred.ndim:
        raise TypeError('targets should have the same shape as pred', ('targets', targets.type, 'pred', pred.type))
    
    # check if targets is of the correct datatype
    if targets.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(pred, targets))

def untuple(a):
    if isinstance(a, tuple):
        return untuple(list(a))
    if isinstance(a, (numpy.ndarray, numpy.generic) ):
        return a
    if isinstance(a, list):
        for i in range(len(a)):
            a[i] = untuple(a[i])
    return a

# allow to specify the size separately -- sometimes an issue when cloning, scanning, etc.
# see the theano-tests/random-streams-scan-clone.py
def dropout(srng, dropout_rate, inp, size=None):
    if size is None:
      size = inp.shape  
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-dropout_rate, size=size, dtype=theano.config.floatX)
    # The cast is important because int * float32 = float64 which pulls things off the gpu
    output = inp * mask
    return output

# "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
# http://arxiv.org/abs/1312.6120
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)


def stopTimer(start, message):
    print message + " took %0.03f seconds" % (time.clock() - start)

def startTimer(message):
    start = time.clock()
    return lambda: stopTimer(start, message)