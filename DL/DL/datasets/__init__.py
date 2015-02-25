
import numpy
import cPickle as pickle
import os
import urllib
import gzip
from ..utils import untuple

datasetPath = '/'.join((__file__.split('/')[:-1]+['']))
 
def getDataset(name, url):
    name = datasetPath + name
    if not os.path.isfile(name):
        print "Retieving dataset from %s" % (url)
        urllib.urlretrieve(url, name)

    if not os.path.isfile(name):
        print "Cannot find dataset %s" % (name)

    if name[-2:] == 'gz':
        f = gzip.open(name, 'rb')
        data = pickle.load(f)
        f.close()
        return data
    else:
        f = open(name, 'rb')
        data = pickle.load(f)
        f.close()
        return data

def mnist():
    dataset = getDataset('mnist.pkl.gz', 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    return untuple(dataset)


def imdb(validation_ratio=0.1, vocabulary_size=10000, maxlen=100):
    """
    validation_ratio: ratio of training data set aside for validation
    vocabulary_size: Vocabulary size. Assuming the larger the word number, 
        the less often it occurs. Unknown words are set to 1
    maxlen: Sequence longer then this get ignored
    """

    train_set = getDataset('imdb.pkl', 'http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl')
    test_set  = getDataset('imdb.pkl', 'http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl')

    # filter out the sequences longer than maxlen
    new_train_set_x = []
    new_train_set_y = []
    for x, y in zip(train_set[0], train_set[1]):
        if len(x) < maxlen:
            new_train_set_x.append(x)
            new_train_set_y.append(y)
    train_set = (new_train_set_x, new_train_set_y)
    del new_train_set_x, new_train_set_y
    

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sample_idx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - validation_ratio)))
    valid_set_x = [train_set_x[s] for s in sample_idx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sample_idx[n_train:]]
    train_set_x = [train_set_x[s] for s in sample_idx[:n_train]]
    train_set_y = [train_set_y[s] for s in sample_idx[:n_train]]
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    # all words outside the vocabulary are set to 1
    removeUnknownWords = lambda x: [[1 if word >= vocabulary_size else word for word in review] for review in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = removeUnknownWords(train_set_x)
    valid_set_x = removeUnknownWords(valid_set_x)
    test_set_x = removeUnknownWords(test_set_x)

    # sort the sequences by their length
    sortLength = lambda sequences: sorted(range(len(sequences)), key=lambda x: len(sequences[x]))

    sorted_index = sortLength(test_set_x)
    test_set_x = [test_set_x[i] for i in sorted_index]
    test_set_y = [test_set_y[i] for i in sorted_index]

    sorted_index = sortLength(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in sorted_index]
    valid_set_y = [valid_set_y[i] for i in sorted_index]

    sorted_index = sortLength(train_set_x)
    train_set_x = [train_set_x[i] for i in sorted_index]
    train_set_y = [train_set_y[i] for i in sorted_index]

    # gather the dataset again
    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    dataset = [train, valid, test]
    return untuple(dataset)

