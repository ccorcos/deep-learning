# Deep Learning

Some deep learning models and experiments built with Theano. 
The bulk of everything interesting is the the `layer.py` file. 
This contains all the classes for hidden layers and different models.

# To Do

- hessian free optimization
  - try on the dbn. not working yet because softmax gradient not implemented
  - try on the rnn, make sure that the linear versions and nonlinear outputs rely on each other in the graph.
- save and load models

state estimation with RNN