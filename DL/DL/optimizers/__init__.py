#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import time
import random
from ..utils import startTimer
import sgd
import rmsprop
import adadelta

optimizers = {
    'sgd': sgd.sgd,
    'rmsprop': rmsprop.rmsprop,
    'adadelta': adadelta.adadelta,
}

def optimize(dataset=None,
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
             test_batches=-1,
             print_cost=False,
             optimizer='rmsprop',
             **options):


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

    print "compiling test function"
    stop = startTimer("compiling test function")
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

    print "compiling validate function"
    stop = startTimer("compiling validate function")
    validate_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens=valid_givens
    )
    stop()

    print "computing gradients"
    stop = startTimer("computing gradients")
    gparams = T.grad(cost, params)
    stop()


    updates = optimizers[optimizer](
        params=params,
        gparams=gparams,
        **options
    )

    print optimizer + ": compiling training function"
    stop = startTimer(optimizer + ": compiling training function")
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

    print optimizer + ": optimizing..."
    try:
        while (epoch < n_epochs) and (not impatient):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                if print_cost:
                    print "    cost: %0.05f" % minibatch_avg_cost
                    
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

    except KeyboardInterrupt:
        print ""
        print ""
        print optimizer + ": optimization interupted"
        print ""

    end_time = time.clock()

    print 'Optimiztation complete'
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print 'Best validation score of %f %% obtained at iteration %i, with test performance %f %%' % (best_validation_loss * 100., best_iter + 1, test_loss * 100.)
    print ""

    train_loss = None
    valid_loss = None
    test_loss = None

    try:
        print "computing model errors"
        print ""
        print "  training..."
        train_losses = [train_model(i) for i in xrange(n_train_batches)]
        train_loss = numpy.mean(train_losses)
        print "  validation..."
        valid_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        valid_loss = numpy.mean(valid_losses)
        print "  test..."
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        test_loss = numpy.mean(test_losses)
        print "\n  train: %0.05f \n  validation: %0.05f \n  test: %0.05f \n" % (train_loss, valid_loss, test_loss)
    
    except KeyboardInterrupt:
        print "computing model errors interrupted"
    
    return train_loss, valid_loss, test_loss
