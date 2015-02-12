#!/usr/bin/python
# coding: utf-8

# import theano
import theano.tensor as T
# import numpy
from ForwardFeed import ForwardFeed
from ..utils import *
import operator

class ForwardFeedColumn(object):
    """ForwardFeedColumn Class

    Multiple parallel ForwardFeeds with column max at the end.
    """

    def __init__(self, rng, input, layer_sizes=[], dropout_rate=0, n_parallel=None, params=None, activation='tanh'):
        """Initialize the parameters for the forward feed

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        layer_sizes: array of ints, dimensionality of each layer size, input to output

        n_parallel: number of parallel forward feeds because they're column-maxed at the end

        activation: string, nonlinearity to be applied in the hidden layer
        """

        ffs = []
        for i in range(n_parallel):
            ff = ForwardFeed(
                rng=rng,
                input=input,
                dropout_rate=dropout_rate, 
                params=maybe(lambda: params[i]),
                layer_sizes=layer_sizes,
                activation=activation)
            ffs.append(ff)

        outputs = map(lambda ff: ff.output, ffs)
        self.output = T.stacklists(outputs).max(axis=0)
        self.ffs = ffs

        self.params = map(lambda x: x.params, self.ffs)

        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.ffs))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.ffs))
        self.updates = reduce(operator.add, map(lambda x: x.updates, self.layers), [])
