#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from utils import *

x = T.vector()
y = T.vector()
z = T.vector()

X = T.stacklists([x,y,z])

M = T.max(X, axis=0)

together = theano.function(inputs=[x,y,z], outputs=X)
maximum = theano.function(inputs=[x,y,z], outputs=M)

print together([1,2,3],[4,5,6], [7,8,9])
print maximum([1,2,3],[4,5,6], [7,8,9])
