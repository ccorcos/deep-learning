#!/usr/bin/python
# coding: utf-8

from USE import USE

class EUSE(USE):
    """Embedded Unsupervised State Estimator

    This is a basic USE but has an embedding layer on the observation input.
    """


    def __init__(self, rng, obs, act, n_obs, e_obs, n_act, n_hidden, dropout_rate=0, ff_obs=[], ff_filt=[], ff_trans=[], ff_act=[], ff_pred=[], activation='tanh', outputActivation='softmax', params=None):
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

        USE.__init__(self, 
            rng=rng, 
            obs=eObs, 
            act=act, 
            n_obs=e_obs, 
            b_act=n_act, 
            n_hidden=n_hidden, 
            dropout_rate=dropout_rate, 
            ff_obs=ff_obs, 
            ff_filt=ff_filt, 
            ff_trans=ff_trans, 
            ff_act=ff_act, 
            ff_pred=ff_pred, 
            activation=activation, 
            outputActivation=outputActivation, 
            params=maybe(lambda: params[1])
        )

        self.layers += [self.embeddingLayer]
        self.params += [self.embeddingLayer.params]
        self.L1 += self.embeddingLayer.L1
        self.L2_sqr += self.embeddingLayer.L2