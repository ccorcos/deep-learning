import theano
import theano.tensor as T
import numpy


# x = T.matrix('x')
# y = T.tensor3('y')

# sx = x.shape
# sy = y.shape

# shape2 = theano.function(inputs=[x], outputs=sx)
# shape3 = theano.function(inputs=[y], outputs=sy)

# print shape2(numpy.random.randn(2,3))
# print shape3(numpy.random.randn(2,3,4))

# raise

x = T.matrix('x')
y = T.tensor3('y')

sx = x.shape
sy = y.shape

sx = T.concatenate([sx[:-1], [10]])
sy = T.concatenate([sy[:-1], [10]])

shape2 = theano.function(inputs=[x], outputs=sx)
shape3 = theano.function(inputs=[y], outputs=sy)

print shape2(numpy.random.randn(2,3))
print shape3(numpy.random.randn(2,3,4))