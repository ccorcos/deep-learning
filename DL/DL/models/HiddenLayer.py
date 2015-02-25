#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, params=None, activation='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal (tanh actually) activation function. Weight matrix W is of shape (n_in, n_out)
        and the bias vector b is of shape (n_out,).

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        n_out: int, number of hidden units

        activation: string, nonlinearity to be applied in the hidden layer
        """
        self.input = input
    
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # [Xavier10] suggests that you should use 4 times larger initial 
        # weights for sigmoid compared to tanh.
        W = None
        b = None
        
        if params is not None:
            W = params[0]
            b = params[1]

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation is 'sigmoid':
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)


        self.output_linear = T.dot(input, W) + b
        self.output = (self.output_linear if activation is None else activations[activation](self.output_linear))

        self.params = [W, b]
        self.weights = [W]

        self.L1 = compute_L1(self.weights)
        self.L2_sqr = compute_L2_sqr(self.weights)