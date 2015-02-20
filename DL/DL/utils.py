#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy

def load_data(dataset, output="int32"):
    ''' Loads the dataset to the GPU

    dataset = [train_set, valid_set, test_set]
    
    each set is a tuple (input, target)
    input is a matrix where rows are a sample
    target is a 1d array of what output should be
    '''

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        Create a shared dataset, copying the whole thing to the GPU.
        We dont want to copy each minibatch over one at a time.
        """
        sharedData = []
        for data in data_xy:
            shared = theano.shared(numpy.asarray(data,
                                                   dtype=theano.config.floatX),
                                                   borrow=borrow)
            sharedData.append(shared)

        # You have to store values on the GPU as floats. But the y is 
        # really an int so we'll cast back to an int for what we return
        if output == 'int32':
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