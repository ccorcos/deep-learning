#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from utils import *

# hide warnings
import warnings
warnings.simplefilter("ignore")

relu = lambda x: T.switch(x<0, 0, x)
cappedrelu =  lambda x: T.minimum(T.switch(x<0, 0, x), 6)
sigmoid = T.nnet.sigmoid
tanh = T.tanh
# softmax = T.nnet.softmax

def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True)) 
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out

activations = {
    'relu': relu,
    'cappedrelu': cappedrelu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'linear': lambda x: x,
    'softmax': softmax
}

"""
Layers are inanimate. They have inputs, outputs, L1, L2_sqr. Thats it. Objects built
on top of layers simply pass these through.
"""



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix W
    and bias vector b. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, params=None):
        """ Initialize the parameters of the logistic regression

        input: theano.tensor, matrix of size (n_examples, n_in)
        n_in: int, number of input units
        n_out: int, number of output units

        """

        W = None
        b = None
        if params is not None:
            W = params[0]
            b = params[1]


        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

        if b is None:
            # initialize the baises b as a vector of n_out 0s
            b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k

        output_linear = T.dot(input, self.W) + self.b
        self.p_y_given_x = softmax(output_linear)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        y: theano.tensor, the correct labels output

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # negative log likelihood, picking of the correct value for each prediction.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        y: theano.tensor, the correct labels output
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

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
       
class ForwardFeed(object):
    """ForwardFeed Class

    This is just a chain of hidden layers.
    """

    def __init__(self, rng, input, layer_sizes=[], dropout_rate=0, params=None, activation='tanh'):
        """Initialize the parameters for the forward feed

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        layer_sizes: array of ints, dimensionality of each layer size, input to output

        activation: string, nonlinearity to be applied in the hidden layer
        """

        output = input
        layers = []
        for i in range(0, len(layer_sizes)-1):
            h = HiddenLayer(
                rng=rng,
                input=output,
                dropout_rate=dropout_rate,
                params=maybe(lambda: params[i]),
                n_in=layer_sizes[i],
                n_out=layer_sizes[i+1],
                activation=activation)
            output = h.output
            layers.append(h)

        self.layers = layers
        self.output = output

        self.params = map(lambda x: x.params, self.layers)

        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

class ForwardFeedColumn(object):
    """ForwardFeedColumn Class

    Multiple parallel ForwardFeeds with column max at the end.
    """

    def __init__(self, rng, input, layer_sizes=[], dropout_rate=0, n_parallel=None, params=None, activation='tanh'):
        """Initialize the parameters for the forward feed

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        layer_sizes: array of ints, dimensionality of each layer size, input to output

        n_parallel: number of parallel forward feeds because they're column-maxed at the end

        activation: string, nonlinearity to be applied in the hidden layer
        """

        ffs = []
        for i in range(n_parallel):
            ff = ForwardFeed(
                rng=rng,
                input=input,
                dropout_rate=dropout_rate, 
                params=maybe(lambda: params[i]),
                layer_sizes=layer_sizes,
                activation=activation)
            ffs.append(ff)

        outputs = map(lambda ff: ff.output, ffs)
        self.output = T.stacklists(outputs).max(axis=0)
        self.ffs = ffs

        self.params = map(lambda x: x.params, self.ffs)

        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.ffs))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.ffs))

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function  while the top layer is a softamx layer.
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_rate=0, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the multilayer perceptron

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        n_hidden: int, number of hidden units

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout in the hidden layer

        activation: string, nonlinearity to be applied in the hidden layer
        """

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            dropout_rate=dropout_rate,
            n_out=n_hidden,
            activation=activation,
            params=maybe(lambda: params[0])
        )

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            dropout_rate=0,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[1])
        )

        self.layers = [self.hiddenLayer, self.outputLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

        self.loss = self.layers[-1].loss
        self.errors = self.layers[-1].errors
        self.output = self.layers[-1].output
        self.pred = self.layers[-1].pred

class DBN(object):
    """Deep Belief Network Class

    A Deep Belief network is a feedforward artificial neural network model
    that has many layers of hidden units and nonlinear activations.
    """

    def __init__(self, rng, input, n_in, n_out, layer_sizes=[], dropout_rate=0, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the multilayer perceptron

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        layer_sizes: array of ints, dimensionality of the hidden layers

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout in the hidden layer

        activation: string, nonlinearity to be applied in the hidden layer
        """

        self.ff = ForwardFeed(
            rng=rng,
            input=input,
            layer_sizes=[n_in] + layer_sizes,
            dropout_rate=dropout_rate,
            activation=activation,
            params=maybe(lambda: params[0])
        )

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.ff.output,
            n_in=layer_sizes[-1],
            dropout_rate=0,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[1])
        )

        self.layers = [self.ff, self.outputLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

        self.loss = self.layers[-1].loss
        self.errors = self.layers[-1].errors
        self.output = self.layers[-1].output
        self.pred = self.layers[-1].pred

class DBNPC(object):
    """A Parallel Column Deep Belief Network Class

    A Parallel Column Deep Belief network has multiple parallel forward feed DBNs
    but pool together their hidden layers into columns at intermitten layers.
    """

    def __init__(self, rng, input, n_in, n_out, ff_sizes=[], n_parallel=1, dropout_rate=0, activation='tanh', outputActivation='softmax', params=None):
        """Initialize the parameters for the multilayer perceptron

        rng: random number generator, e.g. numpy.random.RandomState(1234)

        input: theano.tensor matrix of shape (n_examples, n_in)

        n_in: int, dimensionality of input

        ff_sizes: an array of layer_sizes

        n_parallel: number of parallel DBNs

        n_out: int, number of hidden units

        dropout_rate: float, if dropout_rate is non zero, then we implement a Dropout in the hidden layer

        activation: string, nonlinearity to be applied in the hidden layer
        """

        ffcs = []
        output = input
        for i in range(len(ff_sizes)):
            layer_sizes = ff_sizes[i]
            if i is 0:
                layer_sizes = [n_in] + layer_sizes
            else:
                layer_sizes = [ff_sizes[i-1][-1]] + layer_sizes
            ffc = ForwardFeedColumn(
                rng=rng,
                input=output,
                n_parallel=n_parallel,
                layer_sizes=layer_sizes,
                dropout_rate=dropout_rate,
                activation=activation,
                params=maybe(lambda: params[i])
            )
            output = ffc.output
            ffcs.append(ffc)


        self.outputLayer = HiddenLayer(
            rng=rng,
            input=ffcs[-1].output,
            n_in=ff_sizes[-1][-1],
            dropout_rate=0,
            n_out=n_out,
            activation=outputActivation,
            params=maybe(lambda: params[-1])
        )

        self.ffcs = ffcs

        self.layers = self.ffcs + [self.outputLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)

        self.loss = self.layers[-1].loss
        self.errors = self.layers[-1].errors
        self.output = self.layers[-1].output
        self.pred = self.layers[-1].pred



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