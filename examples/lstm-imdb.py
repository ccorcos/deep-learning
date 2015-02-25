# n_in
# n_emb

dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
vocabulary_size=10000,  # Vocabulary size
validation_ratio=0.05
maxlen=100,  # Sequence longer then this get ignored

dataset = datasets.imdb(validation_ratio=validation_ratio, vocabulary_size=vocabulary_size, maxlen=maxlen)
train = dataset[0]
valid = dataset[1]
test = dataset[2]


# need to process the data and produce a mask so we can handle batches with differnet length sequences!




# dataset has 3 elements, train, validation and test sets
# the first item in each set is a matrix of (n_examples, n_timesteps) with a number representing each word
# the second item is a vector of {0,1} sentiment

# (n_examples, n_timesteps)
x = T.imatrix('x')
mask = T.matrix('mask')

embeddingLayer = EmbeddingLayer(
    rng=rng=rng, 
    input=x, 
    n_in=vocabulary_size, 
    n_out=dim_proj, 
    onehot=False
)

# (n_examples, n_timesteps, dim_proj)
x_emb = embeddingLayer.output

lstm = LSTM(
    rng=rng, 
    input=x_emb, 
    mask=mask, 
    n_units=dim_proj, 
    activation='tanh'
)

y = lstm.output


proj = y

# only get the active and mean mool.
proj = (proj * mask[:, :, None]).sum(axis=0)
proj = proj / mask.sum(axis=0)[:, None]


proj = dropout(srng, dropout_rate, proj)

outputLayer = HiddenLayer(
    rng=rng, 
    input=proj, 
    n_in=dim_proj,
    n_out=2, # sentiment
    params=None, 
    activation='tanh'
)

z = outputLayer.output

# prediction
# get all the params
# coost function, errors
# run rmsprop!


# modify rnn to handle varying length sequences