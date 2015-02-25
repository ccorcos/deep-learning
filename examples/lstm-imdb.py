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
print "    x ---> x_emb ---> LSTM ---> meanPool ---> softmax ---> {1,0} sentiment"
print "loading IMDB"

dim_proj=128               # word embeding dimension and LSTM number of hidden units.
vocabulary_size=10000      # Vocabulary size
maxlen=100                 # Sequence longer then this get ignored
dropout_rate = 0.5
validation_ratio=0.05


# imdb has 3 elements, train, validation and test sets
# the first input in each set is a matrix of (n_examples, n_timesteps) with a number representing each word
# the second input is a vector of {0,1} sentiment
imdb = datasets.imdb(validation_ratio=validation_ratio, vocabulary_size=vocabulary_size, maxlen=maxlen)

# mutate the dataset to pad and mask the sequences
# the sequences are the first input in the dataset
# now each set consists of [padded_sequences, targets, sequence_mask] with shapes:
# [(n_examples, maxlen), (n_examples), (n_examples, maxlen)]
# note that the mask must remain float32!
datasetPadAndMask(imdb, 0)

print "loading data to the GPU"

dataset = load_data(imdb)

print "creating the LSTM"
x = T.matrix('x')          # input words, (n_examples, maxlen)
t = T.vector('t')          # targets
mask = T.matrix('mask')     # mask for valid words (n_examples, maxlen)

inputs = [x, t, mask]       # the mask comes last!

ix = x.astype('int32')      
it = t.astype('int32')         


rng = numpy.random.RandomState(int(time.time())) # random number generator
srng = T.shared_randomstreams.RandomStreams(int(time.time()))

embeddingLayer = EmbeddingLayer(
    rng=rng, 
    input=ix, 
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

# (n_examples, maxlen, dimproj)
z = lstm.output

# only get the active and mean mool.

# mask[:, :, None].shape = (n_examples, maxlen, 1)
# (z * mask[:, :, None]).shape = (n_examples, maxlen, dim_proj)
# (z * mask[:, :, None]).sum(axis=1).shape = (n_examples, dim_proj)
z = (z * mask[:, :, None]).sum(axis=1)

# mask.sum(axis=1).shape = (n_examples,)
# mask.sum(axis=1)[:, None].shape = (n_examples,1)
meanPool = z / mask.sum(axis=1)[:, None]
# meanPool is now (n_examples, dim_proj)

meanPool_drop = dropout(srng, dropout_rate, meanPool)

outputLayer = HiddenLayer(
    rng=rng, 
    input=meanPool_drop, 
    n_in=dim_proj,
    n_out=2,               # {0,1} sentiment
    params=None, 
    activation='softmax'
)

y = outputLayer.output

layers = [embeddingLayer, lstm, outputLayer]

L1 = layers_L1(layers)
L2_sqr = layers_L2_sqr(layers)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    nll_multiclass(y, it)
    + L1_reg * L1
    + L2_reg * L2_sqr
)

pred = pred_multiclass(y)

errors = pred_error(pred, it)

params = flatten(layers_params(layers))

print "training the LSTM with rmsprop"
rmsprop(dataset=dataset,
        inputs=inputs,
        cost=cost,
        params=params,
        errors=errors,
        n_epochs=1000,
        batch_size=100,
        patience=500,
        patience_increase=1.5,
        improvement_threshold=0.995,
        test_batches=1)

print "compiling the prediction function"
predict = theano.function(inputs=[x, mask], outputs=pred)

print "predicting the first 10 samples of the test dataset"
print "predict:", predict(dataset[2][0].get_value()[0:10], dataset[2][-1].get_value()[0:10])
print "answer: ", dataset[2][1].get_value()[0:10]


