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
softmax = T.nnet.softmax

# # you need the bleeding edge Theano to get 'keep_dims'
# def softmax(x):
#     e_x = T.exp(x - x.max(axis=1, keep_dims=True)) 
#     out = e_x / e_x.sum(axis=1, keep_dims=True)
#     return out

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

        self.output_linear = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(self.output_linear)

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

        self.output_linear = T.dot(input, self.W) + self.b
        self.output = (self.output_linear if activation is None else activations[activation](self.output_linear))
       
        if dropout_rate > 0:
            srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-dropout_rate, size=self.output.shape)
            # The cast is important because int * float32 = float64 which pulls things off the gpu
            self.output = self.output * T.cast(mask, theano.config.floatX)

        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()
       
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
        self.output_linear = layers[-1].output_linear

        self.params = map(lambda x: x.params, self.layers)

        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers))

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
        self.output_linear = outputs[T.stacklists(outputs).argmax(axis=0)]
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

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_rate=0, activation='tanh', params=None):
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

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            params=maybe(lambda:params[1])
        )

        self.output_linear = self.logRegressionLayer.output_linear

        self.layers = [self.hiddenLayer, self.logRegressionLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers))

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        self.output = self.logRegressionLayer.y_pred
        self.outputDistribution = self.logRegressionLayer.p_y_given_x

class DBN(object):
    """Deep Belief Network Class

    A Deep Belief network is a feedforward artificial neural network model
    that has many layers of hidden units and nonlinear activations.
    """

    def __init__(self, rng, input, n_in, n_out, layer_sizes=[], dropout_rate=0, activation='tanh', params=None):
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

        self.logRegressionLayer = LogisticRegression(
            input=self.ff.output,
            n_in=layer_sizes[-1],
            n_out=n_out,
            params=maybe(lambda:params[1])
        )

        self.output_linear = self.logRegressionLayer.output_linear

        self.layers = [self.ff, self.logRegressionLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers))

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        self.output = self.logRegressionLayer.y_pred
        self.outputDistribution = self.logRegressionLayer.p_y_given_x

class DBNPC(object):
    """A Parallel Column Deep Belief Network Class

    A Parallel Column Deep Belief network has multiple parallel forward feed DBNs
    but pool together their hidden layers into columns at intermitten layers.
    """

    def __init__(self, rng, input, n_in, n_out, ff_sizes=[], n_parallel=1, dropout_rate=0, activation='tanh', params=None):
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


        self.logRegressionLayer = LogisticRegression(
            input=ffcs[-1].output,
            n_in=ff_sizes[-1][-1],
            n_out=n_out,
            params=maybe(lambda:params[-1])
        )

        self.ffcs = ffcs

        self.output_linear = self.logRegressionLayer.output_linear

        self.layers = self.ffcs + [self.logRegressionLayer]
        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers))

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        self.output = self.logRegressionLayer.y_pred
        self.outputDistribution = self.logRegressionLayer.p_y_given_x



class Recurrence(object):
    """A Reccurence Class which wraps an architecture into a recurrent one."""

    def __init__(self, input, input_t, output_t, recurrent_t, recurrent_tm1, recurrent_0, output_t_linear, recurrent_t_linear):
        """Initialize the recurrence class with the input, output, the recurrent variable
        and the initial recurrent variable.

        This compute in minibatches, so input is (n_examples, n_timesteps, n_in)
        input_t is (n_examples, n_in)

        """

        # compute the recurrence
        def step(x_t, h_tm1):
            h_t_linear = theano.clone(recurrent_t_linear, replace={
                input_t: x_t, 
                recurrent_tm1: h_tm1
            })
            h_t = theano.clone(recurrent_t, replace={
                input_t: x_t, 
                recurrent_tm1: h_tm1
            })
            y_t_linear = theano.clone(output_t_linear, replace={
                recurrent_t: h_t
            })            
            y_t = theano.clone(output_t, replace={
                recurrent_t: h_t
            })
            return h_t, y_t, h_t_linear, y_t_linear

        h0_t = T.extra_ops.repeat(recurrent_0[numpy.newaxis, :], input.shape[0], axis=0)

        [h, y, h_l, y_l], _ = theano.scan(step,
                            sequences=input.dimshuffle(1,0,2), # swap the first two dimensions to scan over n_timesteps
                            outputs_info=[h0_t, None, None, None])

        # swap the dimensions back to (n_examples, n_timesteps, n_out)
        h = h.dimshuffle(1,0,2)
        y = y.dimshuffle(1,0,2)

        h_l = h_l.dimshuffle(1,0,2)
        y_l = y_l.dimshuffle(1,0,2)

        self.output = y
        self.recurrent = h

        self.output_linear = y_l
        self.recurrent_linear = h_l



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
            output_t_linear=self.outputLayer.output_linear,
            recurrent_t_linear=self.hiddenLayer.output_linear
        )

        self.output = R.output
        self.h = R.recurrent

        self.output_linear = R.output_linear
        self.h_linear = R.recurrent_linear

        self.layers = [self.hiddenLayer, self.outputLayer]
        self.params = [self.h0] + map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers))
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers))

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
