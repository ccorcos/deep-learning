#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.models.RNN import RNN
from DL.optimizers.sgd import sgd
from DL.utils import *
import time
import matplotlib.pyplot as plt

# hide warnings
import warnings
warnings.simplefilter("ignore")

print "Testing an RNN with linear outputs"
print "Generating lag test data..."
n_hidden = 30
n_in = 5
n_out = 3
n_steps = 11
n_seq = 100

numpy.random.seed(0)
# simple lag test
seq = numpy.random.randn(n_seq, n_steps, n_in)
targets = numpy.zeros((n_seq, n_steps, n_out))

targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

targets += 0.01 * numpy.random.standard_normal(targets.shape)

# split into training, validation, and test
trainIdx = int(numpy.floor(4./6.*n_seq))
validIdx = int(numpy.floor(5./6.*n_seq))

lagData = ((seq[0:trainIdx,:,:], targets[0:trainIdx,:,:]), 
           (seq[trainIdx:validIdx,:,:], targets[trainIdx:validIdx,:,:]),
           (seq[validIdx:,:,:], targets[validIdx:,:,:]))

print "loading data to the GPU"
dataset = load_data(lagData, output="float32")

print "creating the RNN"
x = T.tensor3('x')  # input
t = T.tensor3('t')  # targets
inputs = [x,t]
rng = numpy.random.RandomState(int(time.time())) # random number generator

rnn = RNN(rng=rng, 
          input=x, 
          n_in=n_in, 
          n_hidden=n_hidden, 
          n_out=n_out, 
          activation='tanh', 
          outputActivation='linear'
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    rnn.loss(t)
    + L1_reg * rnn.L1
    + L2_reg * rnn.L2_sqr
)

errors = rnn.errors(t)
params = flatten(rnn.params)

print "training the rnn with sgdem"

sgd(dataset=dataset,
    inputs=inputs,
    cost=cost,
    params=params,
    errors=errors,
    learning_rate=0.01,
    momentum=0.2,
    n_epochs=5000,
    batch_size=20,
    patience=1000,
    patience_increase=2.,
    improvement_threshold=0.9995)

print "compiling the prediction function"

predict = theano.function(inputs=[x], outputs=rnn.output)

print "predicting the first sample of the training dataset"

fig = plt.figure()
ax1 = plt.subplot(211)
plt.plot(seq[0])
ax1.set_title('input')

ax2 = plt.subplot(212)
true_targets = plt.plot(targets[0])

guess = predict(seq[0:1])[0]
guessed_targets = plt.plot(guess, linestyle='--')
for i, x in enumerate(guessed_targets):
    x.set_color(true_targets[i].get_color())

ax2.set_title('solid: true output, dashed: model output')
plt.show()
