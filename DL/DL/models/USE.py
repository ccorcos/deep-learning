#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from ..utils import *
from ForwardFeed import ForwardFeed
from HiddenLayer import HiddenLayer


class USE(object):
    """ Unserpervised State Estimator Class

    A USE is like an RNN but it is an unsupervised model with actions and observations,
    with the goal of predicting the next observations given all previous actions and
    observations.

    INPUTS:
    n_obs: number of observation inputs
    n_act: number of action inputs
    n_hidden: number of hidden state nodes
    ff_obs: an array of layer sizes for the deep input
    ff_filt: an array of layer sizes for the deep transition
    ff_trans: an array of layer sizes for the deep transition
    ff_act: an array of layer sizes for the deep input
    ff_pred: an array of layer sizes for the deep output
   
    MODEL:
    o_t: observation at time t
    a_t: action at time t
    y_t: is the prediction
    h_t: the state representation
    k_t: the predictive state representation


                          o_t              a_t
                           |                |
                           |                |
               observation |         action |
                           |                |
                           |                |
                filter     ▼   transform    ▼
     k_{t-1} ----------▶  h_t ----------▶  k_t ----------▶  h_{t+1}
                                            |
                                            |
                                  predictor |
                                            |
                                            |
                                            ▼
                                           y_t  
    """

    def __init__(self, rng, obs, act, n_obs, n_act, n_hidden, dropout_rate=0, ff_obs=[], ff_filt=[], ff_trans=[], ff_act=[], ff_pred=[], activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the recurrent neural network state estimator

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        obs, act: theano.tensor matrix of shape (n_examples, n_timesteps, n_in)

        n_obs: int, dimensionality of input observations

        n_act: int, dimensionality of input actions

        n_hidden: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout all hidden layers

        activation: string, nonlinearity to be applied in the hidden layer
        """

        self.obs = obs
        self.act = act

        # create the k0 prior
        k0 = None
        if params:
            k0 = params[0]
        else:
            k0_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_hidden + n_hidden)),
                    high=numpy.sqrt(6. / (n_hidden + n_hidden)),
                    size=(n_hidden,)
                ),
                dtype=theano.config.floatX
            )
            if activation is 'sigmoid':
                k0_values *= 4

            k0 = theano.shared(value=k0_values, name='k0', borrow=True)

        self.k0 = k0

        # Create the computation graph

        self.k_tm1 = T.matrix('k_tm1') # n_examples, n_hidden @ t-1

        self.o_t = T.matrix('o_t') # n_examples, n_obs @ some specific time
        self.a_t = T.matrix('a_t') # n_examples, n_act @ some specific time
       
        self.observationFF = ForwardFeed(
            rng=rng,
            input=self.o_t,
            layer_sizes=[n_obs] + ff_obs,
            params=maybe(lambda: params[1]),
            activation=activation
        )

        self.filterFF = ForwardFeed(
            rng=rng,
            input=self.k_tm1,
            layer_sizes=[n_hidden] + ff_filt,
            params=maybe(lambda: params[2]),
            activation=activation
        )

        self.hLayer = HiddenLayer(
            rng=rng,
            input= T.concatenate([self.filterFF.output, self.observationFF.output], axis=1),
            n_in=([n_obs] + ff_obs)[-1] + ([n_hidden] + ff_filt)[-1],
            dropout_rate=dropout_rate,
            n_out=n_hidden,
            activation=activation,
            params=maybe(lambda: params[3])
        )

        self.h_t = self.hLayer.output

        def obsStep(o_t, k_tm1):
            replaces = {self.o_t: o_t, self.k_tm1: k_tm1}
            h_t = theano.clone(self.h_t, replace=replaces)
            return h_t

        self.transformFF = ForwardFeed(
            rng=rng,
            input=self.h_t,
            layer_sizes=[n_hidden] + ff_trans,
            params=maybe(lambda: params[4]),
            activation=activation
        )

        self.actionFF = ForwardFeed(
            rng=rng,
            input=self.a_t,
            layer_sizes=[n_act] + ff_act,
            params=maybe(lambda: params[5]),
            activation=activation
        )

        self.kLayer = HiddenLayer(
            rng=rng,
            input= T.concatenate([self.transformFF.output, self.actionFF.output], axis=1),
            n_in=([n_hidden] + ff_trans)[-1] + ([n_act] + ff_act)[-1],
            dropout_rate=dropout_rate,
            n_out=n_hidden,
            activation=activation,
            params=maybe(lambda: params[6])
        )

        self.k_t = self.kLayer.output


        self.predictorFF = ForwardFeed(
            rng=rng,
            input=self.kLayer.output,
            layer_sizes=[n_hidden] + ff_pred,
            params=maybe(lambda: params[7]),
            activation=activation
        )

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.predictorFF.output,
            n_in=([n_hidden] + ff_pred)[-1],
            n_out=n_obs,
            activation=outputActivation,
            params=maybe(lambda: params[8])
        )

        self.y_t = self.outputLayer.output
        
        def actStep(a_t, h_t):
            replaces = {self.a_t: a_t, self.h_t: h_t}
            k_t = theano.clone(self.k_t, replace=replaces)
            y_t = theano.clone(self.y_t, replace=replaces)
            return k_t, y_t

        # the first timestep all have the same k0
        k0_t = T.extra_ops.repeat(self.k0[numpy.newaxis, :], obs.shape[0], axis=0)
        # compute the first 
        h0 = obsStep(obs[:,0,:], k0_t)

        # compute the recurrence
        def step(a_t, o_t, h_t):
            k_t, y_t = actStep(a_t, h_t)
            h_tp1 = obsStep(o_t, k_t)
            return h_tp1, k_t, y_t


        [h_t, k_t, y_t], _ = theano.scan(step,
                            sequences=[act.dimshuffle(1,0,2), obs.dimshuffle(1,0,2)], # swap the first two dimensions to scan over n_timesteps
                            outputs_info=[h0, None, None])

        # k0_t is (n_examples, n_hidden)
        # k_t is (n_steps, n_examples, n_hidden)

        k_t = T.concatenate([k0_t[numpy.newaxis,:,:], k_t])
        h_t = T.concatenate([h0[numpy.newaxis,:,:], k_t])

        # swap the dimensions back to (n_examples, n_timesteps, n_out)
        self.h_t = h_t.dimshuffle(1,0,2)
        self.k_t = k_t.dimshuffle(1,0,2)
        self.y_t = y_t.dimshuffle(1,0,2)
        self.output = self.y_t

        self.layers = [self.observationFF, self.filterFF, self.hLayer, self.transformFF, self.actionFF, self.kLayer, self.predictorFF, self.outputLayer]
        self.params = [self.k0] + map(lambda x: x.params, self.layers)
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
            self.obs_pred = T.cast(self.obs[:,1:,:], 'int32')
        elif outputActivation == 'softmax':
            # This is a pain in the ass!
            self.loss = self.nll_multiclass
            self.errors = self.predictionErrors
            self.pred = T.argmax(self.output, axis=-1)
            self.obs_pred = T.argmax(self.obs, axis=-1)[:,1:]

        else:
            raise NotImplementedError


    def mse(self):
        return T.mean((self.output - self.obs[:, 1:, :]) ** 2)

    def nll_binary(self):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.output, self.obs[:, 1:, :]))

    def nll_multiclass(self):
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
        y_f = self.obs_pred.flatten(ndim=1)
        return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])

    def predictionErrors(self):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if self.obs_pred.ndim != self.pred.ndim:
            raise TypeError('y should have the same shape as self.pred',
                ('y', y.type, 'pred', self.pred.type))
        # check if y is of the correct datatype
        if self.obs_pred.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.pred, self.obs_pred))
        else:
            raise NotImplementedError()