#!/usr/bin/python
# coding: utf-8

# import theano
import theano.tensor as T
# import numpy
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# rng = RandomStreams(1234)
# use_noise = theano.shared(numpy_floatX(0.))

def dropout(input, use_noise, rng):
    output = T.switch(use_noise,
                        (input * rng.binomial(input.shape, p=0.5, n=1, dtype=input.dtype)),
                         input * 0.5)
    return output