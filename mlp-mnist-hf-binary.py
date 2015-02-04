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

print "An MLP on MNIST."
print "loading MNIST"
f = open('mnist.pkl', 'rb')
mnist = pickle.load(f)
f.close()

convert2onehot = lambda y: map(lambda x: onehot(x, 10), y)

mnist = list(mnist)
for i in range(3):
    mnist[i] = list(mnist[i])
    mnist[i][1] = numpy.array(convert2onehot(mnist[i][1]))

print "loading data"
gradient_dataset = SequenceDataset(mnist[0], batch_size=None, number_batches=5000)
cg_dataset = SequenceDataset(mnist[0], batch_size=None, number_batches=1000)
valid_dataset = SequenceDataset(mnist[1], batch_size=None, number_batches=1000)

print "creating the MLP"
x = T.matrix('x')  # input
t = T.imatrix('t')  # targets
rng = numpy.random.RandomState(int(time.time())) # random number generator

# construct the MLP class
mlp = MLP(
    rng=rng,
    input=x,
    n_in=28 * 28,
    n_hidden=500,
    n_out=10,
    outputActivation='sigmoid'
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    mlp.loss(t)
    + L1_reg * mlp.L1
    + L2_reg * mlp.L2_sqr
)

errors = mlp.errors(t)
params = list(flatten(mlp.params))

print "training the MLP with hf"
hf = hf_optimizer(
    params, 
    [x, t],                               # array of symbolic variable inputs of your computational graph
    mlp.output_linear,                    # output layer linear activation
    [cost, errors],                       # an array of costs to be optimized
    0.5*(mlp.hiddenLayer.output + 1),     # structural damping, typically the hidden layer, also convert from tanh
    mlp.hiddenLayer.output_linear        # the gauss-newton matrix for the stuctural damping
)

hf.train(gradient_dataset, cg_dataset, initial_lambda=0.5, mu=1.0, preconditioner=False, validation=valid_dataset)


print "compiling the prediction function"
predict = theano.function(inputs=[x], outputs=mlp.pred)

print "predicting the first 10 samples of the test dataset"
print "predict:", numpy.argmax(predict(mnist[2][0][0:10]), axis=1)
print "answer: ", numpy.argmax(mnist[2][1][0:10], axis=1)