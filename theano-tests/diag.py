#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T

x = T.matrix()
y = T.vector()

d = T.diag(x)
z = T.diag(y)


matrixDiags = theano.function(inputs=[x], outputs=d)
vectorDiag = theano.function(inputs=[y], outputs=z)

print matrixDiags([[1,2,3],[4,5,6], [7,8,9]])
print vectorDiag([1,2,3])
