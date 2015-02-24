# Some Deep Learning models built with Theano


## Refactor

Dropout layer is a simple function
Error and loss are separate functions

## To Do

Try using multiplicative RNN!
Various other optimization methods on USE, SSE, EUSE
LSTM
Writing DL.md, https://imgur.com/a/Hqolp
embedding visualization, http://lvdmaaten.github.io/tsne/


## Getting Started

Some examples use `sparkprob` to visualize probablity distributions at the commandline so you may need to install it

    pip install sparkprob

All of the examples use the `DL` package. To use it:
  
    cd DL
    python setup.py develop

To unlink this package when you are done:

    cd DL
    python setup.py develop --uninstall

To load the datasets

    cd datasets
    curl -O http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    curl -O http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl


### Models

The general idea works like this. A "model" take symbolic tensors representing a minibatch. The model constructs the computational graph and produces some class variables to hook into: params, L1, L2_sqr, loss, errors, output, pred. Thus, we can compose models and propagate the L1 and L2_sqr for regularization. We can save the params and pass them as inputs to load a model. We can use the loss and the errors to pass into an optimization function. And we can create a prediction function using output or pred.

Some subtleties here about the naming. Loss is typically something like cross-entropy-error, negative-log-likelihood, or mean-square-error. Errors would be something like, did the model predict the correct MNIST number? Similarly, the output of model could be a distribution of MNIST number likelihoods, but pred, the prediction, is the number with the maximum likelihood.

### Optimizers

An "optimizer" takes in a dataset which is a list of 3 elements: training dataset, validation dataset, test dataset. Each these sub-datasets is an array of data for each of the inputs to the computational graph of the model. Note that the inputs to the "computational graph of the model" includes the "outputs of the model" so-to-speak. The order of the data must correspond to the order of the tensors list passed into inputs. Optimizers are also given param's to update with respect to a cost function. The errors are used for test and validation.






# ToDo

```python




def rmsprop(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update
```