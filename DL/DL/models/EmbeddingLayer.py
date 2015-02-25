#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *

class EmbeddingLayer(object):
    def __init__(self, rng, input, n_in, n_out, onehot=False, params=None):
    
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # [Xavier10] suggests that you should use 4 times larger initial 
        # weights for sigmoid compared to tanh.
        W = None
        if params is not None:
            W = params[0]

        if W is None:
            W_values = numpy.asarray(
                rng.randn(n_in, n_out),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values * 0.01, name='W', borrow=True)

        if onehot:
            self.output = T.dot(input, W)
        else:
            # change the last dimension to the projected dimensino
            shape = T.concatenate([input.shape[:-1], [n_out]])
            self.output = W[input.flatten()].reshape(shape)

        self.params = [W]
        self.L1 = 0
        self.L2_sqr = (W ** 2).sum()