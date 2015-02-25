#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.models.MLP import MLP
from DL.optimizers.sgd import sgd
from DL import datasets
from DL.utils import *
import time

# hide warnings
import warnings
warnings.simplefilter("ignore")

from theano.sandbox.rng_mrg import MRG_RandomStreams


print "An MLP with dropout on MNIST."
print "loading MNIST"
mnist = datasets.mnist()
mnist = untuple(mnist)
print "loading data to the GPU"

# pin the dropout_toggle on the front. leave as float32
datasetWithDropout(mnist)
dataset = load_data(mnist, ["float32", "float32", "int32"])

print "creating the MLP"
x = T.matrix('x')  # input
t = T.ivector('t')  # targets
d = T.vector('dropout_toggle')
inputs = [d, x, t]

rng = numpy.random.RandomState(int(time.time())) # random number generator

# need a trng for dropout
trng = MRG_RandomStreams(int(time.time()))

# construct the MLP class
mlp = MLP(
    rng=rng,
    trng=trng,
    input=x,
    dropout_toggle=d,
    n_in=28 * 28,
    n_hidden=500,
    n_out=10
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    nll_multiclass(mlp.output, t)
    + L1_reg * mlp.L1
    + L2_reg * mlp.L2_sqr
)


pred = pred_multiclass(mlp.output)

errors = pred_error(pred, t)

params = flatten(mlp.params)

print "training the MLP with sgd"
sgd(dataset=dataset,
    inputs=inputs,
    cost=cost,
    params=params,
    errors=errors,
    learning_rate=0.01,
    momentum=0.2,
    n_epochs=1000,
    batch_size=20,
    patience=10000,
    patience_increase=2,
    improvement_threshold=0.995)

print "compiling the prediction function"
predict = theano.function(inputs=[x,d], outputs=pred)

print "predicting the first 10 samples of the test dataset"
print "predict:", predict(mnist[2][0][0:10])
print "answer: ", mnist[2][1][0:10]