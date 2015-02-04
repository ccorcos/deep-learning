#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator
import time


def sgd(dataset=None,
        inputs=None,
        targets=None,
        cost=None,
        params=None,
        errors=None,
        learning_rate=0.01, 
        n_epochs=1000,
        batch_size=20):

    """
    stochastic gradient descent optimization
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

    validate_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens={
            inputs: valid_set_x[index * batch_size:(index + 1) * batch_size],
            targets: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

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

    start_time = time.clock()

    for epoch in range(n_epochs):
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            
        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        print 'epoch %i, validation error %f %%' % (epoch, this_validation_loss * 100.)
                
    end_time = time.clock()

    training_losses = [train_model(i) for i in xrange(n_train_batches)]
    training_loss = numpy.mean(training_losses)

    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
    validation_loss = numpy.mean(validation_losses)

    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_loss = numpy.mean(test_losses)

    print 'Optimiztation complete'
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print 'training:   ' + fmt(training_loss*100.)
    print 'validation: ' + fmt(validation_loss*100.)
    print 'test:       ' + fmt(test_loss*100.)

