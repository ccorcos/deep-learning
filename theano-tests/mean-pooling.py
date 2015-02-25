#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from DL.utils import *


n_examples = 4
maxlen = 3
dim_proj = 2

mask = numpy.random.randn(n_examples, maxlen)
z    = numpy.random.randn(n_examples, maxlen, dim_proj)


# only get the active and mean mool.

# mask[:, :, None].shape = (n_examples, maxlen, 1)
# (z * mask[:, :, None]).shape = (n_examples, maxlen, dim_proj)
# (z * mask[:, :, None]).sum(axis=1).shape = (n_examples, dim_proj)
z = (z * mask[:, :, None]).sum(axis=1)
# mask.sum(axis=1).shape = (n_examples,)
# mask.sum(axis=1)[:, None].shape = (n_examples,1)
z = z / mask.sum(axis=1)[:, None]
# z is now (n_examples, dim_proj)