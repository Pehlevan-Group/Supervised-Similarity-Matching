import cPickle
import gzip
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor.extra_ops

class External_World(object):

	def __init__(self):
		
		dir_path = os.path.dirname(os.path.abspath(__file__))
		print dir_path
		path = dir_path+os.sep+"mnist.pkl.gz"
		print path
		# DOWNLOAD MNIST DATASET
		if not os.path.isfile(path):
			import urllib
			origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
			print 'Downloading data from %s' % origin
			urllib.urlretrieve(origin, path)

		# LOAD MNIST DATASET
		f = gzip.open(path, 'rb')
		print 'File opened'
		(train_x_values, train_y_values), (valid_x_values, valid_y_values), (test_x_values, test_y_values) = cPickle.load(f)
		print 'Loaded'
		f.close()

		# CONCATENATE TRAINING, VALIDATION AND TEST SETS
		x_values = list(train_x_values) + list(valid_x_values) + list(test_x_values)
		y_values = list(train_y_values) + list(valid_y_values) + list(test_y_values)
		self.x =		theano.shared(np.asarray(x_values, dtype=theano.config.floatX), borrow=True)
		self.y = T.cast(theano.shared(np.asarray(y_values, dtype=theano.config.floatX), borrow=True), 'int32')

		self.size_dataset = len(x_values)


class External_World_Normalized(object):

    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        print dir_path
        path = dir_path + os.sep + "mnist.pkl.gz"
        print path
        # DOWNLOAD MNIST DATASET
        if not os.path.isfile(path):
            import urllib
            origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, path)

        # LOAD MNIST DATASET
        f = gzip.open(path, 'rb')
        print 'File opened'
        (train_x_values, train_y_values), (valid_x_values, valid_y_values), (
        test_x_values, test_y_values) = cPickle.load(f)
        print 'Loaded'
        f.close()

        # CONCATENATE TRAINING, VALIDATION AND TEST SETS
        x_values = list(train_x_values) + list(valid_x_values) + list(test_x_values)
        y_values = list(train_y_values) + list(valid_y_values) + list(test_y_values)
        x_values = np.asarray(x_values, dtype=np.float32)
        y_values = np.asarray(y_values, dtype=np.float32)
        x_norm = x_values - np.mean(x_values, axis=1).reshape(-1, 1)
        x_norm = x_norm / (np.linalg.norm(x_norm, axis=1).reshape(-1, 1))
        
        self.x = theano.shared(np.asarray(x_norm, dtype=theano.config.floatX), borrow=True)
        self.y = T.cast(theano.shared(np.asarray(y_values, dtype=theano.config.floatX), borrow=True), 'int32')
        print "Checking normalization, should be 1: ", sum(np.multiply(x_norm[0, :], x_norm[0, :]))

        self.size_dataset = len(x_norm)
