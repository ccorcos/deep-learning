#!/usr/bin/python
# coding: utf-8

# import theano
# import theano.tensor as T
# import numpy
from ForwardFeed import ForwardFeed
from HiddenLayer import HiddenLayer
from ..utils import *

class DBN(object):
    """Deep Belief Network Class

    A Deep Belief network is a feedforward artificial neural network model
    that has many layers of hidden units and nonlinear activations.
    """

    def __init__(self, rng, input, n_in, n_out, layer_sizes=[], dropout_rate=0, srng=None, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the multilayer perceptron

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        layer_sizes: array of ints, dimensionality of the hidden layers

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout in the hidden layer

        activation: string, nonlinearity to be applied in the hidden layer
        """

        ff = ForwardFeed(
            rng=rng,
            input=input,
            layer_sizes=[n_in] + layer_sizes,
            activation=activation,
            params=maybe(lambda: params[0]),
            dropout_rate=dropout_rate, 
            srng=srng,
        )

        outputLayer = HiddenLayer(
            rng=rng,
            input=ff.output,
            n_in=layer_sizes[-1],
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[1])
        )

        self.layers = [ff, outputLayer]

        self.params = layers_params(self.layers)
        self.L1 = layers_L1(self.layers)
        self.L2_sqr = layers_L2_sqr(self.layers)

        self.output = outputLayer.output
