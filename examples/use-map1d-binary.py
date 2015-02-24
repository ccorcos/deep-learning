#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.models.USE import USE
from DL.optimizers.sgd import sgd
from DL.utils import *
import time
from sparkprob import *

# hide warnings
import warnings
warnings.simplefilter("ignore")

print "Testing the USE with binary observation predictions."
print "Generating 1D map test data..."

# inputs are left or right actions.
# the observation is the position on a map with walls on both sides

def squeeze(n, mn, mx):
  if n > mx:
    return mx
  elif n < mn:
    return mn
  else:
    return n

n_obs = 5      # width of the map
n_act = 2      # left, right
n_steps = 20
n_seq = 100

observations = []
actions = []
for i in range(n_seq):
    obs = [int(round(numpy.random.uniform(0, n_obs-1)))]
    act = []
    for j in range(n_steps):
        a = round(numpy.random.uniform())
        d = (a-0.5)*2.0
        act.append(onehot(int(a),2))
        obs.append(squeeze(obs[-1]+int(d), 0, n_obs-1))
    observations.append(map(lambda x: onehot(x, n_obs), obs))
    actions.append(act)

observations = numpy.array(observations)
actions = numpy.array(actions)

# split into training, validation, and test
trainIdx = int(numpy.floor(4./6.*n_seq))
validIdx = int(numpy.floor(5./6.*n_seq))

data = ((observations[0:trainIdx,:,:], actions[0:trainIdx,:,:]), 
        (observations[trainIdx:validIdx,:,:], actions[trainIdx:validIdx,:,:]),
        (observations[validIdx:,:,:], actions[validIdx:,:,:]))

print "loading data to the GPU"
dataset = load_data(data, output="float32")

print "creating the USE"
o = T.tensor3('o')  # observations
a = T.tensor3('a')  # actions
inputs = [o,a]
rng = numpy.random.RandomState(int(time.time())) # random number generator

use = USE(
    rng=rng, 
    obs=o,
    act=a,
    n_obs=n_obs,
    n_act=n_act,
    n_hidden=50,
    activation='relu',
    outputActivation="sigmoid"
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    use.loss()
    + L1_reg * use.L1
    + L2_reg * use.L2_sqr
)

errors = use.errors()
params = flatten(use.params)

print "training the USE with sgdem"

sgd(dataset=dataset,
    inputs=inputs,
    cost=cost,
    params=params,
    errors=errors,
    learning_rate=0.01,
    momentum=0.2,
    n_epochs=5000,
    batch_size=100,
    patience=500,
    patience_increase=2.,
    improvement_threshold=0.9995)


print "compiling the prediction function"

predict = theano.function(inputs=[o, a], outputs=use.output)

print "predicting the first sample of the training dataset"

idx = 2
obs = observations[idx]
act = actions[idx]
y = predict([obs], [act])[0]
print 'obs'.center(len(obs[0])*2) + '  |  ' + 'y'.center(len(y[0])*2) + '  |  ' + 'act'.center(len(y[0])*2)
print ''
print sparkprob(obs[0]) + '  |  ' + sparkprob([0]*6) + '  |  ' +  sparkprob(act[0])
for i in range(1, len(act)):
    print sparkprob(obs[i]) + '  |  ' + sparkprob(y[i-1]) + '  |  ' +  sparkprob(act[i])
print sparkprob(obs[len(obs)-1]) + '  |  ' + sparkprob(y[len(obs)-2]) + '  |  ' + sparkprob([0]*5)
