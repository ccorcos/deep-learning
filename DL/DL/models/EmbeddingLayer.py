#!/usr/bin/python
# coding: utf-8

# import theano
# import theano.tensor as T
# import numpy
# from utils import *
from HiddenLayer import HiddenLayer

class EmbeddingLayer(HiddenLayer):
    """Embedding Class

    This class is basically just a linear hidden layer. There will likely be some additional 
    functionality in this class for visualization layer
    """

    def __init__(self, rng, input, n_in, n_out, params=None):
        """ Initialize the parameters of the embedding layer

        input: theano.tensor, matrix of size (n_examples, n_in)
        n_in: int, number of input units
        n_out: int, number of output units

        """
        HiddenLayer.__init__(self, rng, input, n_in, n_out, dropout_rate=0, params=params, activation='linear')
