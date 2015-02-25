#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.utils import *
from DL import datasets

dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
vocabulary_size=10000,  # Vocabulary size
validation_ratio=0.05
maxlen=100,  # Sequence longer then this get ignored

dataset = datasets.imdb(validation_ratio=validation_ratio, vocabulary_size=vocabulary_size, maxlen=maxlen)
train = dataset[0]
valid = dataset[1]
test = dataset[2]

# dataset has 3 elements, train, validation and test sets
# the first item in each set is a matrix of (n_examples, n_timesteps) with a number representing each word
# the second item is a vector of {0,1} sentiment

