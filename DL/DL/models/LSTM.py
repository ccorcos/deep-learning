#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
from ..utils import *

"""

Generic LSTM Architecture
LSTM [Graves 2012]

x is the input
y is the output
h is the memory cell

g_i is the input gate
c_i is the input candidate
g_f is the forget gate
g_o is the output gete

s is the sigmoid function
f is a nonlinear transfer function


g_i_t = s(W_i * x_t + U_i * y_tm1 + b_i)
c_i_t = f(W_c * x_t + U_c * y_tm1 + b_c)

g_f_t = s(W_f * x_t + U_f * y_tm1 + b_f)
h_t = g_i_t * c_i_t + g_f_t * h_tm1

g_o_t = s(W_o * x_t + U_o * y_tm1 + V_o * h_t + b_o)
y_t = g_o_t * f(h_t)

"""


class LSTM(object):
    """
    A simplified LSTM Layer.
    http://deeplearning.net/tutorial/lstm.html

    For parallelization:
    g_o_t = s(W_o * x_t + U_o * y_tm1 + b_o)

    From the input and the previous output, we can compute the input, output, 
    and forget gates along with the input candidate. Then we can compute the
    hidden memory units and the output.
        
                              h_tm1 ------
                                          X ------------- X --▶ y_t
                          --▶ g_o_t ------                |
              y_tm1 --   |                                |
                      |-----▶ g_i_t --                    |
      x_t ------------   |            X ----- + --▶ h_t --
                         |--▶ c_i_t --        |      
                         |                    |
                          --▶ g_f_t ------    |
                                          X --
                              h_tm1 ------

    """

    def __init__(self, rng, input, mask, n_units, activation='tanh', params=None):
        
        # LSTM weights
        W = None
        U = None
        b = None
        if params is not None:
            W = params[0]
            U = params[1]
            b = params[2]

        if W is None:
            # g_i, g_f, g_o, c_i
            W_values = numpy.concatenate([ortho_weight(n_units),
                                          ortho_weight(n_units),
                                          ortho_weight(n_units),
                                          ortho_weight(n_units)], axis=1)

            W = theano.shared(value=W_values, name='W', borrow=True)

        if U is None:
            U_values = numpy.concatenate([ortho_weight(n_units),
                                          ortho_weight(n_units),
                                          ortho_weight(n_units),
                                          ortho_weight(n_units)], axis=1)

            U = theano.shared(value=U_values, name='U', borrow=True)

        if b is None:
            b_values = numpy.zeros((4 * n_units,)).astype(theano.config.floatX)

            b = theano.shared(value=b_values, name='b', borrow=True)


        # cut out the gates after parallel matrix multiplication
        def cut(x, n, dim):
            return x[:, :, n * dim:(n + 1) * dim]

        f = activations[activation]
        s = activations['sigmoid']
        def step(mask_t, xWb_t, y_tm1, h_tm1):
            pre_activation = tensor.dot(y_tm1, U) + xWb_t

            g_i_t = s(cut(pre_activation, 0, n_units))
            c_i_t = f(cut(pre_activation, 3, n_units))

            g_f_t = s(cut(pre_activation, 1, n_units))
            h_t = g_f_t * h_tm1 + g_i_t * c_i_t
            # dropout
            h_t = mask_t[:, None] * h_t + (1. - mask_t)[:, None] * h_tm1

            g_o_t = s(cut(pre_activation, 2, n_units))
            y_t = g_o_t * f(h_t)
            # dropout
            y_t = mask_t[:, None] * y_t + (1. - mask_t)[:, None] * y_tm1

            return y_t, h_t


        # timesteps, samples, dimension
        n_timesteps = input.shape[0]
        n_samples = input.shape[1]

        # efficiently compute  the input gate, forget gate, 
        xWb = tensor.dot(input, W) + b

        [y, h], updates = theano.scan(step,
                                    sequences=[mask, xWb],
                                    outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, n_units),
                                                  tensor.alloc(numpy_floatX(0.), n_samples, n_units)])
                                    # n_steps=n_timesteps)

        self.params = [U, W, b]
        self.weights = [U, W]
        self.L1 = compute_L1(self.weights)
        self.L2_sqr = compute_L2_sqr(self.weights)

        self.output = y
