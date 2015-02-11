#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from ..utils import *
from HiddenLayer import HiddenLayer


class Recurrence(object):
    """A Reccurence Class which wraps an architecture into a recurrent one."""

    def __init__(self, input, input_t, output_t, recurrent_t, recurrent_tm1, recurrent_0):
        """Initialize the recurrence class with the input, output, the recurrent variable
        and the initial recurrent variable.

        This compute in minibatches, so input is (n_examples, n_timesteps, n_in)
        input_t is (n_examples, n_in)

        """

        # compute the recurrence
        def step(x_t, h_tm1):
            h_t = theano.clone(recurrent_t, replace={
                input_t: x_t, 
                recurrent_tm1: h_tm1
            })       
            y_t = theano.clone(output_t, replace={
                recurrent_t: h_t
            })
            return h_t, y_t

        h0_t = T.extra_ops.repeat(recurrent_0[numpy.newaxis, :], input.shape[0], axis=0)

        [h, y], _ = theano.scan(step,
                            sequences=input.dimshuffle(1,0,2), # swap the first two dimensions to scan over n_timesteps
                            outputs_info=[h0_t, None])

        # swap the dimensions back to (n_examples, n_timesteps, n_out)
        h = h.dimshuffle(1,0,2)
        y = y.dimshuffle(1,0,2)

        self.output = y
        self.recurrent = h



class RNN(object):
    """Recurrent Neural Network Class

    A RNN looks a lot like an MLP but the hidden layer is recurrent, so the hidden
    layer receives the input and the itselft at the previous time step. RNNs can have
    "deep" transition, inputs, and outputs like so:


     (n_in) ----▶  (n_hidden) ----▶ (n_out)
                 ▲            |
                 |            |
                 |            |
                 -----{t-1}----     
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_rate=0, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the recurrent neural network

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_timesteps, n_in)

        n_in: int, dimensionality of input

        n_hidden: int, number of hidden units

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout all hidden layers

        activation: string, nonlinearity to be applied in the hidden layer
        """

        # create the h0 prior
        h0 = None
        if params:
            h0 = params[0]
        else:
            h0_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_hidden,)
                ),
                dtype=theano.config.floatX
            )
            if activation is 'sigmoid':
                h0_values *= 4

            h0 = theano.shared(value=h0_values, name='h0', borrow=True)

        self.h0 = h0

        # Create the computation graph
        h_tm1 = T.matrix('h_tm1') # n_examples, n_hidden @ t-1
        x_t = T.matrix('x_t') # n_examples, n_in @ some specific time
       
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input= T.concatenate([x_t, h_tm1], axis=1),
            n_in=n_in+n_hidden,
            dropout_rate=dropout_rate,
            n_out=n_hidden,
            activation=activation,
            params=maybe(lambda: params[1])
        )

        h_t = self.hiddenLayer.output

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=h_t,
            n_in=n_hidden,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[1])
        )

        y_t = self.outputLayer.output


        R = Recurrence(
            input=input, 
            input_t=x_t, 
            output_t=y_t, 
            recurrent_t=h_t, 
            recurrent_tm1=h_tm1, 
            recurrent_0=self.h0,
        )

        self.output = R.output
        self.h = R.recurrent

        self.layers = [self.hiddenLayer, self.outputLayer]
        self.params = [self.h0] + map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

        if outputActivation == 'linear':
            self.loss = self.mse
            self.errors = self.mse
        elif outputActivation == 'sigmoid':
            # I was having trouble here with ints and floats. Good question what exactly was going on.
            # I decided to settle with floats and use MSE. Seems to work after all...
            self.loss = self.nll_binary
            self.errors = self.predictionErrors
            self.pred = T.round(self.output)  # round to {0,1}
        elif outputActivation == 'softmax':
            # This is a pain in the ass!
            self.loss = self.nll_multiclass
            self.errors = self.predictionErrors
            self.pred = T.argmax(self.output, axis=-1)
        else:
            raise NotImplementedError


    def mse(self, y):
        # error between output and target
        return T.mean((self.output - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.output, y))

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        #
        # Theano's advanced indexing is limited
        # therefore we reshape our n_steps x n_seq x n_classes tensor3 of probs
        # to a (n_steps * n_seq) x n_classes matrix of probs
        # so that we can use advanced indexing (i.e. get the probs which
        # correspond to the true class)
        # the labels y also must be flattened when we do this to use the
        # advanced indexing
        p_y = self.output
        p_y_m = T.reshape(p_y, (p_y.shape[0] * p_y.shape[1], -1))
        y_f = y.flatten(ndim=1)
        return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])

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
