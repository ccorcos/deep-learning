#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
from layer import *
from utils import *
# from sgd import *
# from sgde import *
from sgdem import *
import cPickle as pickle
import warnings
import time

warnings.simplefilter("ignore")

print "An parallel column DBN on MNIST."
print "loading MNIST"
f = open('mnist.pkl', 'rb')
mnist = pickle.load(f)
f.close()

print "loading data to the GPU"
dataset = load_data(mnist)

print "creating the DBN"
x = T.matrix('x')  # input
t = T.ivector('t')  # targets
rng = numpy.random.RandomState(int(time.time())) # random number generator

# construct the DBN class
dbn = DBNPC(
    rng=rng,
    input=x,
    n_in=28 * 28,
    dropout_rate=0.5,
    ff_sizes=[[10,10,100], [10,10,100]],
    n_parallel=10,
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

print "training the dbn with sgdem"

sgdem(dataset=dataset,
    inputs=x,
    targets=t,
    cost=cost,
    params=params,
    errors=errors,
    learning_rate=0.01,
    momentum=0.2,
    n_epochs=1000,
    batch_size=50,
    patience=10000,
    patience_increase=1.25,
    improvement_threshold=0.995)

print "compiling the prediction function"

predict = theano.function(inputs=[x], outputs=dbn.pred)
distribution = theano.function(inputs=[x], outputs=dbn.output)


print "predicting the first 10 samples of the test dataset"
print "predict:", predict(mnist[2][0][0:10])
print "answer: ", mnist[2][1][0:10]

print "the output distribution should be slightly different each time due to dropout"
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])
print "distribution:", distribution(mnist[2][0][0:1])