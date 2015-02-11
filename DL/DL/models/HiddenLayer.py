#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, dropout_rate=0, params=None, activation='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal (tanh actually) activation function. Weight matrix W is of shape (n_in, n_out)
        and the bias vector b is of shape (n_out,).

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        n_out: int, number of hidden units

        activation: string, nonlinearity to be applied in the hidden layer
        """
        self.input = input
    
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # [Xavier10] suggests that you should use 4 times larger initial 
        # weights for sigmoid compared to tanh.
        W = None
        b = None
        if params is not None:
            W = params[0]
            b = params[1]

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation is 'sigmoid':
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        output_linear = T.dot(input, self.W) + self.b
        output = (output_linear if activation is None else activations[activation](output_linear))
       
        if dropout_rate > 0:
            srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-dropout_rate, size=output.shape)
            # The cast is important because int * float32 = float64 which pulls things off the gpu
            output = output * T.cast(mask, theano.config.floatX)

        self.output = output
        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()

        if activation == 'linear':
            self.loss = self.mse
            self.errors = self.mse
            self.pred = self.output
        elif activation == 'sigmoid':
            # I was having trouble here with ints and floats. Good question what exactly was going on.
            # I decided to settle with floats and use MSE. Seems to work after all...
            self.loss = self.nll_binary
            self.errors = self.predictionErrors
            self.pred = T.round(self.output)  # round to {0,1}
        elif activation == 'softmax':
            # This is a pain in the ass!
            self.loss = self.nll_multiclass
            self.errors = self.predictionErrors
            self.pred = T.argmax(self.output, axis=-1)
        else:
            pass
            # raise NotImplementedError


    def mse(self, y):
        # error between output and target
        return T.mean((self.output - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.output, y))

    def nll_multiclass(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def predictionErrors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.pred.ndim:
            raise TypeError('y should have the same shape as self.pred',
                ('y', y.type, 'pred', self.pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.pred, y))
        else:
            raise NotImplementedError()