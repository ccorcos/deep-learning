#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.models.LSTM import LSTM
from DL.models.EmbeddingLayer import EmbeddingLayer
from DL.models.HiddenLayer import HiddenLayer
from DL.optimizers.rmsprop import rmsprop
from DL import datasets
from DL.utils import *
import time

# hide warnings
import warnings
warnings.simplefilter("ignore")


print "An LSTM with mean-pooling and embedded words on IMDB for sentiment analysis."
print "    x ---> x_emb ---> LSTM ---> mean() ---> softmax ---> {1,0} sentiment"
print "loading IMDB"

dim_proj=128,               # word embeding dimension and LSTM number of hidden units.
vocabulary_size=10000,      # Vocabulary size
maxlen=100,                 # Sequence longer then this get ignored
validation_ratio=0.05


# imdb has 3 elements, train, validation and test sets
# the first input in each set is a matrix of (n_examples, n_timesteps) with a number representing each word
# the second input is a vector of {0,1} sentiment
imdb = datasets.imdb(validation_ratio=validation_ratio, vocabulary_size=vocabulary_size, maxlen=maxlen)



# need to process the data and produce a mask so we can handle batches with differnet length sequences!
# transform the dataset to have a mask!
# LSTM with and without mask
# basically just mask helper functions







print "loading data to the GPU"
# should I try int64?
dataset = load_data(mnist, ["int32", "int32"])


print "creating the LSTM"
x = T.imatrix('x')          # input words
mask = T.matrix('mask')     # mask for valid words
t = T.ivector('t')          # targets
inputs = [x, mask, t]
rng = numpy.random.RandomState(int(time.time())) # random number generator



# (n_examples, n_timesteps)
x = T.imatrix('x')
mask = T.matrix('mask')

embeddingLayer = EmbeddingLayer(
    rng=rng, 
    input=x, 
    n_in=vocabulary_size, 
    n_out=dim_proj, 
    onehot=False
)

# (n_examples, n_timesteps, dim_proj)
x_emb = embeddingLayer.output

lstm = LSTM(
    rng=rng, 
    input=x_emb, 
    mask=mask, 
    n_units=dim_proj, 
    activation='tanh'
)

z = lstm.output

# only get the active and mean mool.
z = (z * mask[:, :, None]).sum(axis=0)
z = z / mask.sum(axis=0)[:, None]


z_drop = dropout(srng, dropout_rate, z)

outputLayer = HiddenLayer(
    rng=rng, 
    input=z_drop, 
    n_in=dim_proj,
    n_out=2,          # {0,1} sentiment
    params=None, 
    activation='softmax'
)

y = outputLayer.output

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    nll_multiclass(y, t)
    + L1_reg * mlp.L1
    + L2_reg * mlp.L2_sqr
)

pred = pred_multiclass(y)

errors = pred_error(pred, t)

params = flatten(embeddingLayer.params + lstm.params + outputLayer.params)

print "training the LSTM with rmsprop"
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

# print "predicting the first 10 samples of the test dataset"
# print "predict:", predict(mnist[2][0][0:10])
# print "answer: ", mnist[2][1][0:10]


