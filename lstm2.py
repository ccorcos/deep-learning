"""

LSTM Architecture
http://deeplearning.net/tutorial/lstm.html

x is the input
y is the output
h is the memory cell

g_i is the input gate
c_i is the input candidate
g_f is the forget gate
g_o is the output gete

s is the sigmoid function
f is a nonlinear transfer function

LSTM [Graves 2012]

g_i_t = s(W_i * x_t + U_i * y_tm1 + b_i)
c_i_t = f(W_c * x_t + U_c * y_tm1 + b_c)

g_f_t = s(W_f * x_t + U_f * y_tm1 + b_f)
h_t = g_i_t * c_i_t + g_f_t * h_tm1

g_o_t = s(W_o * x_t + U_o * y_tm1 + V_o * h_t + b_o)
y_t = g_o_t * f(h_t)

Simplification for parallelization:
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

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)



















def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model="",  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):



    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    vocabulary_size=10000,  # Vocabulary size
    validation_ratio=0.05
    maxlen=100,  # Sequence longer then this get ignored



    dataset = datasets.imdb(validation_ratio=validation_ratio, vocabulary_size=vocabulary_size, maxlen=maxlen)
    train = dataset[0]
    valid = dataset[1]
    test = dataset[2]


    load_data, prepare_data = (imdb.load_data, imdb.prepare_data)

    train, valid, test = load_data(n_words=n_words, , maxlen=maxlen)
 
    ydim = numpy.max(train[1]) + 1


    print 'Building model'
    # embedding
    randn = numpy.random.rand(vocabulary_size, dim_proj)
    Wemb = (0.01 * randn).astype(theano.config.floatX)


    # LSTM weights
    lstmW = numpy.concatenate([ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj)], axis=1)
    lstmU = numpy.concatenate([ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj)], axis=1)
    lstmb = numpy.zeros((4 * dim_proj,)).astype(theano.config.floatX)



    # Classifier weights
    U = 0.01 * numpy.random.randn(dim_proj,ydim).astype(theano.config.floatX)
    b = numpy.zeros((ydim,)).astype(theano.config.floatX)


    # make params shared
    Wemb, lstmW, lstmU, lstmb, U, b
    theano.shared(params[kk], name=kk)
