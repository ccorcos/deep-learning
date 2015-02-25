#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *

class EmbeddingLayer(object):
    def __init__(self, rng, input, n_in, n_out, params=None):
        """
        
        """

        self.input = input
    
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # [Xavier10] suggests that you should use 4 times larger initial 
        # weights for sigmoid compared to tanh.
        W = None
        if params is not None:
            W = params[0]

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W
        self.output = T.dot(input, self.W)
       
        self.params = [self.W]
        self.L1 = 0
        self.L2_sqr = (self.W ** 2).sum()