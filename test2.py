import theano
import theano.tensor as T
import numpy
import time

# Make a simple hidden layer:
n_in = 10
n_out = 20
dropout_rate = 0.5

rng = numpy.random.RandomState(int(time.time())) # random number generator
srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))


input = T.matrix('input')

W_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)
    ),
    dtype=theano.config.floatX
)

W = theano.shared(value=W_values, name='W', borrow=True)

b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
b = theano.shared(value=b_values, name='b', borrow=True)

output = T.tanh(T.dot(input, W) + b)


if dropout_rate > 0:
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = T.imatrix('mask')
    # The cast is important because int * float32 = float64 which pulls things off the gpu
    output = output * T.cast(mask, theano.config.floatX)


x0 = T.matrix('x0')
x = T.tensor3('x')

def step(x_t, x_tm1):
    x_tp1 = theano.clone(output, replace={input:x_t, mask:T.cast(srng.binomial(n=1, p=1-dropout_rate, size=output.shape), 'int32')})
    return x_tp1 + x_tm1

z, _ = theano.scan(step, sequences=[x], outputs_info=[x0])

print z