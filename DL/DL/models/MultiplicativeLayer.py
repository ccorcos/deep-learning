#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *
import operator

class MultiplicativeLayer(object):
    def __init__(self, rng, input, mult, n_in, n_out, n_mult, n_fact, params=None, activation='tanh'):
        """
        A Multplicative Tensor Layer using the factorization specified in:

        "2011 Generating Text with Recurrent Neural Networks - Ilya Sutskevers"
        
                          mult  
                            |    
                            |    
                            |    
                   ---------|    
                   |        |    
                   |        ▼    
        input ---- ▲ ---▶  output 


        Generic Multiplicative Unit using Factorization
        x = mult
        h = input
        y = output

        f = diag( W_fx * x ) * ( W_fh * h )
        y = tanh( W_hf * f + W_hx * x + b)


        f = diag( W_fm * mult ) * ( W_fi * input )
        output = tanh( W_of * f + W_om * mult + b)

        rng: random number generator, e.g. numpy.random.RandomState(1234)
        input: theano.tensor matrix of shape (n_examples, n_in)
        mult:  theano.tensor matrix of shape (n_examples, n_mult)

        n_in: int, dimensionality of input
        n_mult: int, dimensionality of the multiplicative input
        n_out: int, number of hidden units
        n_fact: int, the dimensionality of the factorization

        activation: string, nonlinearity to be applied in the hidden layer
        """

        self.input = input
        self.mult = mult
    
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # [Xavier10] suggests that you should use 4 times larger initial 
        # weights for sigmoid compared to tanh.
        W_fi = None
        W_of = None
        W_fm = None
        W_om = None
        b = None

        if params is not None:
            W_fi = params[0]
            W_of = params[1]
            W_fm = params[2]
            W_om = params[3]
            b = params[4]

        def makeWeight(d1, d2, name):
          W_values = numpy.asarray(
              rng.uniform(
                  low=-numpy.sqrt(6. / (d1 + d2)),
                  high=numpy.sqrt(6. / (d1 + d2)),
                  size=(d1, d2)
              ),
              dtype=theano.config.floatX
          )
          if activation is 'sigmoid':
              W_values *= 4

          W = theano.shared(value=W_values, name=name, borrow=True)
          return W

        if W_fi is None:
          W_fi = makeWeight(n_fact, n_in, 'W_fi')
        if W_of is None:
          W_of = makeWeight(n_out, n_fact, 'W_of')
        if W_fm is None:
          W_fm = makeWeight(n_fact, n_mult, 'W_fm')
        if W_om is None:
          W_om = makeWeight(n_out, n_mult, 'W_om')
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W_fi = W_fi
        self.W_of = W_of
        self.W_fm = W_fm
        self.W_om = W_om
        self.b = b

        f = T.dot( diag(T.dot(W_fm, mult)), T.dot(W_fi, input) )
        output_linear = T.dot(W_of, f) + T.dot(W_om, mult) + b
        output = (output_linear if activation is None else activations[activation](output_linear))

        self.output = output
        self.params = [self.W_fi, self.W_of, self.W_fm, self.W_om, self.b]
        weights = [self.W_fi, self.W_of, self.W_fm, self.W_om]
        self.L1 = reduce(operator.add, map(lambda x: abs(x).sum(), weights), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: (x ** 2).sum(), weights), 0)