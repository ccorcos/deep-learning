#!/usr/bin/python
# coding: utf-8

# import theano
# import theano.tensor as T
# import numpy
from HiddenLayer import HiddenLayer
import operator
from ..utils import *

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function  while the top layer is a softamx layer.
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_rate=0, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the multilayer perceptron

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        n_hidden: int, number of hidden units

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout in the hidden layer

        activation: string, nonlinearity to be applied in the hidden layer
        """

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            dropout_rate=dropout_rate,
            n_out=n_hidden,
            activation=activation,
            params=maybe(lambda: params[0])
        )

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            dropout_rate=0,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[1])
        )

        self.layers = [self.hiddenLayer, self.outputLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

        self.loss = self.layers[-1].loss
        self.errors = self.layers[-1].errors
        self.output = self.layers[-1].output
        self.pred = self.layers[-1].pred