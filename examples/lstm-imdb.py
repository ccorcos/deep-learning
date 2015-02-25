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
# should I try int64?
# note that the mask must remain float32!
dataset = load_data(imdb, ["float64", "float64", "float64"])

print "creating the LSTM"
x = T.matrix('x')          # input words, (n_examples, maxlen)
t = T.vector('t')          # targets
mask = T.matrix('mask')     # mask for valid words (n_examples, maxlen)

inputs = [x, t, mask]       # the mask comes last!

x = x.astype('int32')      
t = t.astype('int32')         


rng = numpy.random.RandomState(int(time.time())) # random number generator
srng = T.shared_randomstreams.RandomStreams(int(time.time()))

print "dataset types"

print dataset[0][0].type
print dataset[0][1].type
print dataset[0][2].type

print 'theano.config.floatX', theano.config.floatX
print 'imdb x', x, x.type
print 'imdb t', t, t.type
print 'imdb mask', mask, mask.type

embeddingLayer = EmbeddingLayer(
    rng=rng, 
    input=x, 
    n_in=vocabulary_size, 
    n_out=dim_proj, 
    onehot=False
)

# (n_examples, n_timesteps, dim_proj)
x_emb = embeddingLayer.output

print 'imdb x_emb', x_emb, x_emb.type


lstm = LSTM(
    rng=rng, 
    input=x_emb, 
    mask=mask, 
    n_units=dim_proj, 
    activation='tanh'
)

# (n_examples, maxlen, dimproj)
z = lstm.output

print 'imdb z', z, z.type

# only get the active and mean mool.

# mask[:, :, None].shape = (n_examples, maxlen, 1)
# (z * mask[:, :, None]).shape = (n_examples, maxlen, dim_proj)
# (z * mask[:, :, None]).sum(axis=1).shape = (n_examples, dim_proj)
z = (z * mask[:, :, None]).sum(axis=1)
print 'imdb z', z, z.type

# mask.sum(axis=1).shape = (n_examples,)
# mask.sum(axis=1)[:, None].shape = (n_examples,1)
meanPool = z / mask.sum(axis=1)[:, None]
# meanPool is now (n_examples, dim_proj)

print 'imdb meanPool', meanPool, meanPool.type

meanPool_drop = dropout(srng, dropout_rate, meanPool)


print 'imdb meanPool_drop', meanPool_drop, meanPool_drop.type

outputLayer = HiddenLayer(
    rng=rng, 
    input=meanPool_drop, 
    n_in=dim_proj,
    n_out=2,               # {0,1} sentiment
    params=None, 
    activation='softmax'
)

y = outputLayer.output

print 'imdb y', y, y.type

layers = [embeddingLayer, lstm, outputLayer]

L1 = layers_L1(layers)
L2_sqr = layers_L2_sqr(layers)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    nll_multiclass(y, t)
    + L1_reg * L1
    + L2_reg * L2_sqr
)

print 'nll_multiclass(y, t)', nll_multiclass(y, t).type

pred = pred_multiclass(y)

print 'pred', pred.type

errors = pred_error(pred, t)

print 'errors', errors.type

# theano.printing.debugprint(errors, print_type=True)

params = flatten(layers_params(layers))

print "training the LSTM with rmsprop"
rmsprop(dataset=dataset,
        inputs=inputs,
        cost=cost,
        params=params,
        errors=errors,
        n_epochs=1000,
        batch_size=100,
        patience=5000,
        patience_increase=1.5,
        improvement_threshold=0.995)

print "compiling the prediction function"
predict = theano.function(inputs=[x, mask], outputs=pred)

print "predicting the first 10 samples of the test dataset"
print "predict:", predict(dataset[2][0][0:10], dataset[2][-1][0:10])
print "answer: ", dataset[2][1][0:10]


