#!/usr/bin/python
# coding: utf-8

# import theano
# import theano.tensor as T
# import numpy
# from utils import *
from HiddenLayer import HiddenLayer

class LogisticRegression(HiddenLayer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix W
    and bias vector b. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, rng, input, n_in, n_out, params=None):
        """ Initialize the parameters of the logistic regression

        input: theano.tensor, matrix of size (n_examples, n_in)
        n_in: int, number of input units
        n_out: int, number of output units

        """
        HiddenLayer.__init__(self, rng, input, n_in, n_out, params=params, activation='sigmoid')
