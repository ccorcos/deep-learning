#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy

def rmsprop(params, gparams):

    # http://deeplearning.net/tutorial/code/lstm.py
    zipped_grads = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_grads = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gparams)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, gparams)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, gparams)]

    updir = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(params, updir_new)]
    
    updates = zgup + rgup + rg2up + updir_new + param_up

    return updates