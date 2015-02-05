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
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                               borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                               borrow=borrow)
        # You have to store values on the GPU as floats. But the y is 
        # really an int so we'll cast back to an int for what we return
        if output == 'int32':
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(dataset[2])
    valid_set_x, valid_set_y = shared_dataset(dataset[1])
    train_set_x, train_set_y = shared_dataset(dataset[0])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def maybe(func):
    res = None
    try:
        res = func()
    except:
        pass
    return res

def flatten(container):
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten(i):
                yield j
        else:
            yield i

fmt = lambda x: "{:12.8f}".format(x)

def onehot(value, length):
    v = [0]*length
    v[value] = 1
    return v