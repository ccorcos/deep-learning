# Some Deep Learning models built with Theano

## To Do

LSTM. Dropout RNN. Embedding layer class.
USE with dropout, learn embedding layer for one die. do it again for the same die with a different initialization. Do it for a different die. compare the embedding matrices. How do we do this in real time?
Try this on a 2d SLAM dataset.
  - embedding visualization http://lvdmaaten.github.io/tsne/


## Getting Started

Some examples use `sparkprob` to visualize probablity distributions at the commandline so you may need to install it

    pip install sparkprob

All of the examples use the `DL` package. To use it:
  
    cd DL
    python setup.py develop

To unlink this package when you are done:

    cd DL
    python setup.py develop --uninstall

### Models

The general idea works like this. A "model" take symbolic tensors representing a minibatch. The model constructs the computational graph and produces some class variables to hook into: params, L1, L2_sqr, loss, errors, output, pred. Thus, we can compose models and propagate the L1 and L2_sqr for regularization. We can save the params and pass them as inputs to load a model. We can use the loss and the errors to pass into an optimization function. And we can create a prediction function using output or pred.

Some subtleties here about the naming. Loss is typically something like cross-entropy-error, negative-log-likelihood, or mean-square-error. Errors would be something like, did the model predict the correct MNIST number? Similarly, the output of model could be a distribution of MNIST number likelihoods, but pred, the prediction, is the number with the maximum likelihood.

### Optimizers

An "optimizer" takes in a dataset which is a list of 3 elements: training dataset, validation dataset, test dataset. Each these sub-datasets is an array of data for each of the inputs to the computational graph of the model. Note that the inputs to the "computational graph of the model" includes the "outputs of the model" so-to-speak. The order of the data must correspond to the order of the tensors list passed into inputs. Optimizers are also given param's to update with respect to a cost function. The errors are used for test and validation.
