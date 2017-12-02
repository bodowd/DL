"""
Utility functions for data processing and forward propagation for ResNets

"""
import numpy as np
import tensorflow as tf
import h5py
import math

def load_dataset():
    """
    The data is a bunch of images of hands in the shape of numbers (1 finger for the number one, 2 fingers for two etc)

    """
    train_dataset = h5py.File('datasets/train_signs.h5', 'r')
    # training set features
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    # training set labels
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('datasets/test_signs.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    # the classes
    classes = np.array(test_dataset['list_classes'][:])

    # reshape train and test sets
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    """
    # number of training examples
    m = X.shape[0] 
    mini_batches = []
    np.random.seed(seed)

    # shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # partition (shuffled_X, shuffled_Y). Subtract the end case
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of minibatches determined by mini_batch_size in the partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :] 
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # handling the end case where last mini_batch < mini_batch_size
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T # np.eye returns a 2-D identity array
    return Y

def forward_propagation_for_predict(X, parameters):
    """
    Implements forward prop for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input_size, number of examples)
    parameters -- python dict containing your parameters "W1", "b1", "W2", "b2", "W3", "b3" the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit

    """
    # Retrieve parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1) # takes dot product of W1, X and adds the bias term b1
    A1 = tf.nn.relu(Z1) # first layer activations
    Z2 = tf.add(tf.matmul(W2, A1), b2) # 2nd layer
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3) # 3rd layer linear output. Will pass this to a softmax unit in another func

    return Z3

def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters['W1'])
    b1 = tf.convert_to_tensor(parameters['b1'])
    W2 = tf.convert_to_tensor(parameters['W2'])
    b2 = tf.convert_to_tensor(parameters['b2'])
    W3 = tf.convert_to_tensor(parameters['W3'])
    b3 = tf.convert_to_tensor(parameters['b3'])

    params = {'W1': W1,
              'b1': b1,
              'W2': W2,
              'b2': b2,
              'W3': W3,
              'b3': b3}

    x = tf.placeholder('float', [12288,1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    predictions = sess.run(p, feed_dict = {x:X}) # feed_dict {tf.placeholder (x) : input (X)}
    
    return predictions
