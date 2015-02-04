#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from layer import *
from utils import *
from hf import hf_optimizer, SequenceDataset
import cPickle as pickle
import warnings
import time

warnings.simplefilter("ignore")

print "An DBN on MNIST with dropout trained with HF!"
print "loading MNIST"
f = open('mnist.pkl', 'rb')
mnist = pickle.load(f)
f.close()

print "loading data"
gradient_dataset = SequenceDataset(mnist[0], batch_size=None, number_batches=5000)
cg_dataset = SequenceDataset(mnist[0], batch_size=None, number_batches=1000)
valid_dataset = SequenceDataset(mnist[1], batch_size=None, number_batches=1000)

print "creating the DBN"
x = T.matrix('x')  # input
t = T.ivector('t')  # targets
rng = numpy.random.RandomState(int(time.time())) # random number generator

# construct the DBN class
dbn = DBN(
    rng=rng,
    input=x,
    n_in=28 * 28,
    dropout_rate=0.5,
    layer_sizes=[200,200,200],
    activation='tanh',
    n_out=10
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    dbn.loss(t)
    + L1_reg * dbn.L1
    + L2_reg * dbn.L2_sqr
)

errors = dbn.errors(t)
params = list(flatten(dbn.params))

print "training the dbn with hf"

hf = hf_optimizer(
    params, 
    [x, t],                               # array of symbolic variable inputs of your computational graph
    dbn.output_linear,                    # output layer linear activation
    [cost, errors],                       # an array of costs to be optimized
    0.5*(dbn.ff.layers[1].output + 1),    # structural damping, typically the hidden layer, also convert from tanh
    dbn.ff.layers[1].output_linear        # the gauss-newton matrix for the stuctural damping
)

hf.train(gradient_dataset, cg_dataset, initial_lambda=0.5, mu=1.0, preconditioner=False, validation=valid_dataset)

print "compiling the prediction function"

predict = theano.function(inputs=[x], outputs=dbn.output)
distribution = theano.function(inputs=[x], outputs=dbn.outputDistribution)


print "predicting the first 10 samples of the test dataset"
print "predict:", predict(mnist[2][0][0:10])
print "answer: ", mnist[2][1][0:10]

print "the output distribution should be slightly different each time due to dropout"
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])