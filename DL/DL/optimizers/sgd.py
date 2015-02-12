#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import time

def sgd(dataset=None,
        inputs=None,
        cost=None,
        params=None,
        errors=None,
        learning_rate=0.01, 
        momentum=0.1,
        n_epochs=1000,
        batch_size=20,
        patience=10000,
        patience_increase=2,
        improvement_threshold=0.995,
        updates=[]):

    """
    stochastic gradient descent optimization with early stopping and momentum

    for vanilla gradient decent, set the patience to numpy.inf and momentum to 0

    early stopping criteria
    patience: look as this many examples regardless
    patience_increase: wait this much longer when a new best is found
    improvement_threshold: a relative improvement of this much is considered significant

    dataset is a list or tuple of length 3 including the training set, validation set
    and the test set. In each set, these must be a list or tuple of the inputs to
    the computational graph in the same order as the list of Theano.tensor variable
    that are passed in as inputs. The inputs to the graph must accept minibatches meaning
    that the first dimension is the number of training examples. 

    """

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

    print "sgdem: compiling test function"
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

    print "sgdem: compiling validate function"
    validate_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens=valid_givens
    )

    print "sgdem: computing gradients"
    gparams = [T.grad(cost, param) for param in params]
    momentums = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)) for param in params]
    updates = []
    for param, gparam, mom in zip(params, gparams, momentums):
        update = momentum * mom - learning_rate * gparam
        updates.append((mom, update))
        updates.append((param, param + update))

    print "sgdem: compiling training function"
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens=train_givens
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