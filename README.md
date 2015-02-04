# Deep Learning

Some deep learning models and experiments built with Theano. 
The bulk of everything interesting is the the `layer.py` file. 
This contains all the classes for hidden layers and different models.

# Getting Started

Install Theano

    sudo pip install Theano

Make sure you have the MNIST dataset for some of the examples:

    curl -O http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

# To Do

- hessian free optimization
  - this isnt working on the mlp right now -- seems to be an issue with SequenceDataset. This is so much slower to optimize though, so maybe we'll just use sgd for now.
  - try on the dbn. not working yet because softmax gradient not implemented
  - try on the rnn, make sure that the linear versions and nonlinear outputs rely on each other in the graph.
- save and load models

state estimation with RNN