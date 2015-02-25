#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import time
from ..utils import startTimer
import random

def rmsprop(dataset=None,
            inputs=None,
            cost=None,
            params=None,
            errors=None,
            n_epochs=1000,
            batch_size=20,
            patience=10000,
            patience_increase=2,
            improvement_threshold=0.995,
            updates=[],
            test_batches=-1):


    # index to a [mini]batch
    index = T.lscalar()  

    train_set = dataset[0]
    valid_set = dataset[1]
    test_set = dataset[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set[0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set[0].get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set[0].get_value(borrow=True).shape[0] / batch_size

    if n_train_batches == 0:
        n_train_batches = 1
    if n_valid_batches == 0:
        n_valid_batches = 1
    if n_test_batches == 0:
        n_test_batches = 1

    print "rmsprop: compiling test function"
    stop = startTimer("rmsprop: compiling test function")
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_givens = list(updates)
    valid_givens = list(updates)
    train_givens = list(updates)
    for i in range(len(inputs)):
        test_givens.append((inputs[i], test_set[i][index * batch_size:(index + 1) * batch_size]))
        valid_givens.append((inputs[i], valid_set[i][index * batch_size:(index + 1) * batch_size]))
        train_givens.append((inputs[i], train_set[i][index * batch_size:(index + 1) * batch_size]))

    test_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens=test_givens
    )
    stop()

    print "rmsprop: compiling validate function"
    stop = startTimer("rmsprop: compiling validate function")
    validate_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens=valid_givens
    )
    stop()
    
    print "rmsprop: computing gradients"
    stop = startTimer("rmsprop: computing gradients")
    gparams = T.grad(cost, params)
    stop()

    # http://deeplearning.net/tutorial/code/lstm.py
    zipped_grads = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_grads = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gparams)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, gparams)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, gparams)]

    updir = [theano.shared(p.get_value() * numpy.asarray(0., dtype=theano.config.floatX)) for p in params]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(params, updir_new)]
    
    updates = zgup + rgup + rg2up + updir_new + param_up

    print "rmsprop: compiling training function"
    stop = startTimer("rmsprop: compiling training function")
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens=train_givens
    )
    stop()

    validation_frequency = min(n_train_batches, patience / 2)

    start_time = time.clock()

    best_validation_loss = numpy.inf
    best_iter = 0
    test_loss = 0.

    epoch = 0
    impatient = False

    while (epoch < n_epochs) and (not impatient):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            # keep track of how many minibatches we've trained. Every so often, do a validation.
            iteration = (epoch - 1) * n_train_batches + minibatch_index
            if (iteration + 1) % validation_frequency is 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print 'epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)
                print '   iteration %i, patience %i' % (iteration, patience)

                # if we have a better validation then keep track of it
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iteration * patience_increase)
                    # keep track of the best validation
                    best_validation_loss = this_validation_loss
                    best_iter = iteration
                    # remember the test loss as well
                    test_losses = None
                    if test_batches > 0:
                        test_losses = [test_model(i) for i in random.sample(range(n_test_batches), test_batches)]
                    else:
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_loss = numpy.mean(test_losses)
                    print '     epoch %i, minibatch %i/%i, best test error %f %%' % (epoch, minibatch_index + 1, n_train_batches, test_loss * 100.)

            if patience <= iteration:
                impatient = True
                break
                
    end_time = time.clock()

    print 'Optimiztation complete'
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print 'Best validation score of %f %% obtained at iteration %i, with test performance %f %%' % (best_validation_loss * 100., best_iter + 1, test_loss * 100.)









