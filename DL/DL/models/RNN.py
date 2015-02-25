#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *
from HiddenLayer import HiddenLayer


class Recurrence(object):
    """A Reccurence Class which wraps an architecture into a recurrent one."""

    def __init__(self, input, input_t, output_t, recurrent_t, recurrent_tm1, recurrent_0, updates=[]):
        """Initialize the recurrence class with the input, output, the recurrent variable
        and the initial recurrent variable.

        This compute in minibatches, so input is (n_examples, n_timesteps, n_in)
        input_t is (n_examples, n_in)

        """

        # compute the recurrence
        def step(x_t, h_tm1):
            h_t = theano.clone(recurrent_t, replace=updates + [(input_t, x_t), (recurrent_tm1, h_tm1)])       
            y_t = theano.clone(output_t, replace=updates + [(recurrent_t, h_t)])
            return h_t, y_t

        h0_t = T.extra_ops.repeat(recurrent_0[numpy.newaxis, :], input.shape[0], axis=0)

        [h, y], scanUpdates = theano.scan(step,
                            sequences=[input.dimshuffle(1,0,2),], # swap the first two dimensions to scan over n_timesteps
                            outputs_info=[h0_t, None])

        # swap the dimensions back to (n_examples, n_timesteps, n_out)
        h = h.dimshuffle(1,0,2)
        y = y.dimshuffle(1,0,2)

        self.output = y
        self.recurrent = h
        self.updates = scanUpdates



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

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_rate=0, srng=None, activation='tanh', outputActivation='softmax', params=None):
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



        # Create the computation graph
        h_tm1 = T.matrix('h_tm1') # n_examples, n_hidden @ t-1
        x_t = T.matrix('x_t') # n_examples, n_in @ some specific time
       
        hiddenLayer = HiddenLayer(
            rng=rng,
            input= T.concatenate([x_t, h_tm1], axis=1),
            n_in=n_in+n_hidden,
            n_out=n_hidden,
            activation=activation,
            params=maybe(lambda: params[1])
        )

        h_t = hiddenLayer.output

        if dropout_rate > 0:
            assert(srng is not None)
            h_t = dropout(srng, dropout_rate, h_t)

        outputLayer = HiddenLayer(
            rng=rng,
            input=h_t,
            n_in=n_hidden,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[1])
        )

        y_t = outputLayer.output

        self.layers = [hiddenLayer, outputLayer]
        self.params = [h0] + layers_params(self.layers)
        self.L1 = layers_L1(self.layers)
        self.L2_sqr = layers_L2_sqr(self.layers)

        recurrence = Recurrence(
            input=input, 
            input_t=x_t, 
            output_t=y_t, 
            recurrent_t=h_t, 
            recurrent_tm1=h_tm1, 
            recurrent_0=h0,
            updates=srng.updates() if srng else []
        )

        self.output = recurrence.output
        self.h = recurrence.recurrent
        self.updates = recurrence.updates
