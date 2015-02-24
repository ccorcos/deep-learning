#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy


y = T.vector()
n_x = 2
n_y = 3
W_xy = numpy.random.randn(n_x, n_y)
z = T.dot(W_xy, y)

dot = theano.function(inputs=[y], outputs=z)

print dot(numpy.random.randn(n_y))


q = T.vector()
w = T.vector()

vecDot = theano.function(inputs=[q,w], outputs=T.dot(q,w))
vecDot2 = theano.function(inputs=[q,w], outputs=T.dot(w,q))
vecMult = theano.function(inputs=[q,w], outputs=w*q)
vecMult2 = theano.function(inputs=[q,w], outputs=q*w)


print vecDot([1,2], [2,3])
print vecDot2([1,2], [2,3])
print vecMult([1,2], [2,3])
print vecMult2([1,2], [2,3])
