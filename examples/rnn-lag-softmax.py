#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.models.RNN import RNN
from DL.optimizers.rmsprop import rmsprop
from DL.utils import *
import time
import matplotlib.pyplot as plt

# hide warnings
import warnings
warnings.simplefilter("ignore")

print "Testing an RNN with softmax outputs"
print "Generating lag test data..."

n_hidden = 100
n_in = 5
n_steps = 10
n_seq = 100
n_classes = 3
n_out = n_classes  # restricted to single softmax per time step

# simple lag test
seq = numpy.random.randn(n_seq, n_steps, n_in)
targets = numpy.zeros((n_seq, n_steps), dtype=numpy.int)

thresh = 0.5
# if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
# class 1
# if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
# class 2
# if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
# class 0
targets[:, 2:][seq[:, 1:-1, 3] > seq[:, :-2, 0] + thresh] = 1
targets[:, 2:][seq[:, 1:-1, 3] < seq[:, :-2, 0] - thresh] = 2
#targets[:, 2:, 0] = numpy.cast[numpy.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

# split into training, validation, and test
trainIdx = int(numpy.floor(4./6.*n_seq))
validIdx = int(numpy.floor(5./6.*n_seq))

lagData = ((seq[0:trainIdx,:,:], targets[0:trainIdx,:]), 
           (seq[trainIdx:validIdx,:,:], targets[trainIdx:validIdx,:]),
           (seq[validIdx:,:,:], targets[validIdx:,:]))

print "loading data to the GPU"
dataset = load_data(lagData, types=["float32", "int32"])

print "creating the RNN"
x = T.tensor3('x')  # input
t = T.imatrix('t')  # targets
inputs = [x,t]
rng = numpy.random.RandomState(int(time.time())) # random number generator

rnn = RNN(rng=rng, 
          input=x, 
          n_in=n_in, 
          n_hidden=n_hidden, 
          n_out=n_out, 
          activation='tanh', 
          outputActivation='softmax'
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    nll_multiclass_timeseries(rnn.output, t)
    + L1_reg * rnn.L1
    + L2_reg * rnn.L2_sqr
)

pred = pred_multiclass(rnn.output)

errors = pred_error(pred, t)

params = flatten(rnn.params)

print "training the rnn with rmsprop"

rmsprop(dataset=dataset,
        inputs=inputs,
        cost=cost,
        params=params,
        errors=errors,
        n_epochs=5000,
        batch_size=100,
        patience=1000,
        patience_increase=2.,
        improvement_threshold=0.9995)

print "compiling the prediction function"

predict = theano.function(inputs=[x], outputs=rnn.output)

print "predicting the first 10 samples of the training dataset"

seqs = xrange(10)
for seq_num in seqs:
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[seq_num])
    ax1.set_title('input')
    ax2 = plt.subplot(212)

    # blue line will represent true classes
    true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')

    # show probabilities (in b/w) output by model
    guess = predict(seq[seq_num:seq_num+1])[0]
    guessed_probs = plt.imshow(guess.T, interpolation='nearest', cmap='gray')
    ax2.set_title('blue: true class, grayscale: probs assigned by model')

plt.show()