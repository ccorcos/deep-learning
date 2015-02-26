#!/usr/bin/python
# coding: utf-8

import theano
# import theano.tensor as T
import numpy

def sgd(params, gparams,learning_rate=0.01, momentum=0.1):

    """
    stochastic gradient descent optimization with early stopping and momentum

    for vanilla gradient decent, set the patience to numpy.inf and momentum to 0

    early stopping criteria
    patience: look as this many examples regardless
    patience_increase: wait this much longer when a new best is found
    improvement_threshold: a relative improvement of this much is considered significant

    dataset is a list or tuple of length 3 including the training set, validation set
    and the test set. In each set, these must be a list or tuple of the inputs to
    the computational graph in the same order as the list of Theano.tensor variable
    that are passed in as inputs. The inputs to the graph must accept minibatches meaning
    that the first dimension is the number of training examples. 

    """


    momentums = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)) for param in params]
    updates = []
    for param, gparam, mom in zip(params, gparams, momentums):
        update = momentum * mom - learning_rate * gparam
        updates.append((mom, update))
        updates.append((param, param + update))

    return updates