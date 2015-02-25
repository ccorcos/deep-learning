#!/usr/bin/python
# coding: utf-8

# import theano
# import theano.tensor as T
# import numpy
from HiddenLayer import HiddenLayer
from ..utils import *

class ForwardFeed(object):
    """ForwardFeed Class

    This is just a chain of hidden layers.
    """

    def __init__(self, rng, input, layer_sizes=[], dropout_rate=0, srng=None, params=None, activation='tanh'):
        """Initialize the parameters for the forward feed

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        layer_sizes: array of ints, dimensionality of each layer size, input to output

        activation: string, nonlinearity to be applied in the hidden layer
        """

        output = input
        layers = []
        for i in range(0, len(layer_sizes)-1):
            hiddenLayer = HiddenLayer(
                rng=rng,
                input=output,
                params=maybe(lambda: params[i]),
                n_in=layer_sizes[i],
                n_out=layer_sizes[i+1],
                activation=activation)

            h = hiddenLayer.output
            if dropout_rate > 0:
                assert(srng is not None)
                h = dropout(srng, dropout_rate, h)

            output = h
            layers.append(hiddenLayer)

        self.layers = layers
        self.output = output

        self.params = layers_params(self.layers)
        self.L1 = layers_L1(self.layers)
        self.L2_sqr = layers_L2_sqr(self.layers)