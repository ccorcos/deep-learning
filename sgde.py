#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
import time

def sgde(dataset=None,
        inputs=None,
        targets=None,
        cost=None,
        params=None,
        errors=None,
        learning_rate=0.01, 
        n_epochs=1000,
        batch_size=20,
        patience=10000,
        patience_increase=2,
        improvement_threshold=0.995):

    """
    stochastic gradient descent optimization with early stopping

    early stopping criteria
    patience: look as this many examples regardless
    patience_increase: wait this much longer when a new best is found
    improvement_threshold: a relative improvement of this much is considered significant
    """

    # index to a [mini]batch
    index = T.lscalar()  

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "sgde: compiling test function"
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens={
            inputs: test_set_x[index * batch_size:(index + 1) * batch_size],
            targets: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print "sgde: compiling validate function"
    validate_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens={
            inputs: valid_set_x[index * batch_size:(index + 1) * batch_size],
            targets: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print "sgde: computing gradients"
    gparams = [T.grad(cost, param) for param in params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

    print "sgde: compiling training function"
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            inputs: train_set_x[index * batch_size: (index + 1) * batch_size],
            targets: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

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