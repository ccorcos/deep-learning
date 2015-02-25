#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.models.MLP import MLP
from DL.optimizers.rmsprop import rmsprop
from DL import datasets
from DL.utils import *
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# hide warnings
import warnings
warnings.simplefilter("ignore")


print "An MLP with dropout on MNIST."
print "loading MNIST"
mnist = datasets.mnist()

print "loading data to the GPU"
dataset = load_data(mnist, ["float32", "int32"])

print "creating the MLP"
x = T.matrix('x')  # input
t = T.ivector('t')  # targets
inputs = [x, t]
rng = numpy.random.RandomState(int(time.time())) # random number generator
srng = RandomStreams(int(time.time()))

# construct the MLP class
mlp = MLP(
    rng=rng,
    input=x,
    n_in=28 * 28,
    n_hidden=500,
    n_out=10,
    dropout_rate=0.5,
    srng=srng
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

print "training the MLP with rmsprop"
rmsprop(dataset=dataset,
        inputs=inputs,
        cost=cost,
        params=params,
        errors=errors,
        n_epochs=1000,
        batch_size=20,
        patience=5000,
        patience_increase=1.5,
        improvement_threshold=0.995)

print "compiling the prediction function"
predict = theano.function(inputs=[x], outputs=pred)
distribution = theano.function(inputs=[x], outputs=mlp.output)

print "predicting the first 10 samples of the test dataset"
print "predict:", predict(mnist[2][0][0:10])
print "answer: ", mnist[2][1][0:10]

print "with dropout, the output distributions should all be slightly different"
print "predict:", distribution(mnist[2][0][0:1])
print "predict:", distribution(mnist[2][0][0:1])
print "predict:", distribution(mnist[2][0][0:1])
print "predict:", distribution(mnist[2][0][0:1])