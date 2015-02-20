#!/usr/bin/python
# coding: utf-8

from USE import USE
from EmbeddingLayer import *
from ..utils import *
import operator
# import theano
import theano.tensor as T

class EUSE(object):
    """Embedded Unsupervised State Estimator

    This is a basic USE but has an embedding layer on the observation input.
    """

    def __init__(self, rng, obs, act, n_obs, e_obs, n_act, n_hidden, dropout_rate=0, ff_obs=[], ff_filt=[], ff_trans=[], ff_act=[], ff_pred=[], activation='tanh', outputActivation='sigmoid', params=None):
        """
        Same as the USE but has e_obs with is the observation embedding dimension.
        """
        self.embeddingLayer = EmbeddingLayer(
            rng=rng, 
            input=obs, 
            n_in=n_obs, 
            n_out=e_obs, 
            params=maybe(lambda: params[0])
        )

        eObs = self.embeddingLayer.output

        self.use = USE(
            rng=rng, 
            obs=eObs, 
            act=act, 
            n_obs=e_obs, 
            n_act=n_act, 
            n_hidden=n_hidden, 
            dropout_rate=dropout_rate, 
            ff_obs=ff_obs, 
            ff_filt=ff_filt, 
            ff_trans=ff_trans, 
            ff_act=ff_act, 
            ff_pred=ff_pred, 
            activation=activation, 
            outputActivation='sigmoid', 
            params=maybe(lambda: params[1])
        )

        # un embed back to binary
        self.unEmbeddingLayer = UnEmbeddingLayer(
            input=self.use.output,
            Wemb=self.embeddingLayer.W,
            activation=outputActivation
        )

        self.layers = [self.embeddingLayer, self.use, self.unEmbeddingLayer]
        self.output = self.unEmbeddingLayer.output
        self.loss = lambda: self.unEmbeddingLayer.loss(obs[:, 1:, :])
        self.errors = lambda: self.unEmbeddingLayer.errors(T.cast(obs[:,1:,:], 'int32'))

        self.params = map(lambda x: x.params, self.layers)
        self.L1 = reduce(operator.add, map(lambda x: x.L1, self.layers), 0)
        self.L2_sqr = reduce(operator.add, map(lambda x: x.L2_sqr, self.layers), 0)
        self.updates = reduce(operator.add, map(lambda x: x.updates, self.layers), [])