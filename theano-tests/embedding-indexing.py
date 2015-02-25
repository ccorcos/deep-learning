#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.utils import *

input = T.imatrix('input')

n_in = 5
n_out = 4

W = theano.shared(numpy.random.randn(n_in, n_out))

theano.printing.debugprint(input.shape[:-1], print_type=True)

shape = T.concatenate([input.shape, [n_out]])
  
theano.printing.debugprint(shape, print_type=True)


output = W[input.flatten()].reshape(shape, ndim=3)

r = theano.function(inputs=[input], outputs=output)

s = theano.function(inputs=[input], outputs=shape)

# 1 example
# 2 timesteps
# ints representing
print r([
  [0,1,3,2,4], 
  [0,1,3,2,4]
])