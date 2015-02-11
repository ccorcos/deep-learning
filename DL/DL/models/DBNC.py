#!/usr/bin/python
# coding: utf-8

# import theano
# import theano.tensor as T
# import numpy
from ForwardFeedColumn import ForwardFeedColumn
from HiddenLayer import HiddenLayer
import operator
from ..utils import *

class DBNC(object):
    """A Column Deep Belief Network Class

    A Column Deep Belief network has multiple parallel forward feed DBNs
    but pool together their hidden layers into columns at intermitten layers.
    """

    def __init__(self, rng, input, n_in, n_out, ff_sizes=[], n_parallel=1, dropout_rate=0, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the multilayer perceptron

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        ff_sizes: an array of layer_sizes

        n_parallel: number of parallel DBNs

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout in the hidden layer

        activation: string, nonlinearity to be applied in the hidden layer
        """

        ffcs = []
        output = input
        for i in range(len(ff_sizes)):
            layer_sizes = ff_sizes[i]
            if i is 0:
                layer_sizes = [n_in] + layer_sizes
            else:
                layer_sizes = [ff_sizes[i-1][-1]] + layer_sizes
            ffc = ForwardFeedColumn(
                rng=rng,
                input=output,
                n_parallel=n_parallel,
                layer_sizes=layer_sizes,
                dropout_rate=dropout_rate,
                activation=activation,
                params=maybe(lambda: params[i])
            )
            output = ffc.output
            ffcs.append(ffc)


        self.outputLayer = HiddenLayer(
            rng=rng,
            input=ffcs[-1].output,
            n_in=ff_sizes[-1][-1],
            dropout_rate=0,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[-1])
        )

        self.ffcs = ffcs

        self.layers = self.ffcs + [self.outputLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

        self.loss = self.layers[-1].loss
        self.errors = self.layers[-1].errors
        self.output = self.layers[-1].output
        self.pred = self.layers[-1].pred