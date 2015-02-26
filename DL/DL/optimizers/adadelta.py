#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy

def adadelta(params, gparams):

    # http://deeplearning.net/tutorial/code/lstm.py
    zipped_grads = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_up2 = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gparams)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, gparams)]

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    updates = zgup + rg2up + ru2up + param_up

    return updates
