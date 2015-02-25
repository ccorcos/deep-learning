#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from DL.utils import *
from DL import datasets

dim_proj=128  # word embeding dimension and LSTM number of hidden units.
vocabulary_size=10000  # Vocabulary size
validation_ratio=0.05
maxlen=100  # Sequence longer then this get ignored

imdb = datasets.imdb(validation_ratio=validation_ratio, vocabulary_size=vocabulary_size, maxlen=maxlen)
datasetPadAndMask(imdb, 0)
