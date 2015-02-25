import theano
import theano.tensor as T
import numpy
import time

srng = T.shared_randomstreams.RandomStreams(int(time.time()))


input = T.vector('input')
inputs = T.matrix('inputs')


def dropout(srng, dropout_rate, inp, size=None):
    if size is None:
      size = inp.shape  
    # mask = srng.binomial(n=1, p=1-dropout_rate, size=(10,), dtype=theano.config.floatX)
    mask = srng.binomial(n=1, p=1-dropout_rate, size=size, dtype=theano.config.floatX)
    out = inp * mask
    return out

output = dropout(srng, 0.5, input, inputs[0].shape)

def step(x_t):
    upd = [(input, x_t)]
    y_t = theano.clone(output, replace=upd)
    return y_t

outputs, _ = theano.scan(step, sequences=inputs, outputs_info=[None])

predict = theano.function(inputs=[inputs], outputs=outputs)

print predict(numpy.random.randn(2,10))

g = theano.function(inputs=[inputs, outputs], outputs=T.grad(outputs, inputs))

print g

# # define tensor variables
# X = T.matrix("X")
# W = T.matrix("W")
# b_sym = T.vector("b_sym")

# # define shared random stream
# trng = T.shared_randomstreams.RandomStreams(1234)
# d=trng.binomial(size=W[1].shape)

# results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym) * d, sequences=X)
# compute_with_bnoise = theano.function(inputs=[X, W, b_sym], outputs=[results],
#                           updates=updates, allow_input_downcast=True)
# x = numpy.eye(10, 2, dtype=theano.config.floatX)
# w = numpy.ones((2, 2), dtype=theano.config.floatX)
# b = numpy.ones((2), dtype=theano.config.floatX)

# print compute_with_bnoise(x, w, b)