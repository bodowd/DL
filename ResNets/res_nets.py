"""
Reimplementation of ResNets introduced by He et al.

Very deep networks can represent very complex functions, and learn features at different levels of abstraction. A challenge with very deep networks are vanishing gradients. These are when the gradient quickly goes to zero, and then gradient descent becomes too slow to use. As backpropagation moves from the final layer back to the first, the weight matrix is multiplied at each step, and can decrease exponentially to zero or it may explode to very large values.

ResNets offer a solution to this problem by using a "skip connection" that allows the gradient to be directly backprogated back to the earlier layers. 
Normally, activations a[l] = g(z[l])
With skip connections, the activations of a previous layer are added to the activations in a late rlayer like a[l] = g(z[l]+a[l-2]) = g(W[l]*a[l-1] + b[l] + a[l-2])

Two blocks will be constructed:
1. The identity block: standard block used in ResNets. When the input activation has the same dimension as the output activation
2. The convolution block. When the input and output dimensions don't match up. Here there is a convolution in the shortcut path. This conv2d layer is used just to resize the input x to a different dimension so that the shortcut value can be added back to the main path. (ex: to reduce activation dimensions's height and width by a factor of 2, a 1x1 convolution with stride = 2 can be used). No non-linear activation function is used on the conv2d in the shortcut path. 

"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from resnets_utils import *

def identity_block(X, f, filters, stage, block):
    """
    There will be a main path, and a shortcut path(the skip connection).
    The components of the main path are:
    1. Conv2d with 1x1 filters, stride = 1, padding = no padding ('valid' in keras). This is followed by a BatchNorm to normalize the channels axis, and then a ReLU activation function
    2. Conv2d with fxf filters, stride = 1, padding = retain same length as input ('same' in keras), followed by BatchNorm to normalize the channels axis, and then a ReLU activation function.
    3. Conv2d with 1x1 filters and stride = 1, with no padding ('valid'), followed by a BatchNorm. No ReLU applied here.

    Then , the shortcut path is added to the input.
    Finally, a ReLU activation function is applied to the result of that.

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, number of filters in the CONV layers of the main path. The dimensionality of the output space (the number output of filters in the convolution)
    stage -- integer, used for naming the layers
    block -- string/character, used for naming the layers

    Returns:
    X -- tensor of shape (n_H, n_W, n_C). Output of the identity block.

    """
    # defining names bases so that it doesn't need to be typed again and again
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # retrieve filters
    F1, F2, F3 = filters

    # save the input value so that it can be added back to the main path (this is the shortcut path)
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    # Second component of the main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)

    # Final step: Add shortcut value to main path and pass through ReLU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block.

    Main path components:
    1. Conv2D with F1 filters of shape (1,1) and stride (s,s). No padding ('valid' in keras).
    Followed by BatchNorm to normalize across channels axis. Then applying ReLU activation function
    2. Conv2D with F2 filters of shape (f,f) and stride (1,1). Padding same. Followed by BatchNorm to normalize across channel axis. Then ReLU activation function is applied.
    3. Conv2D with F3 filters of (1,1) and stride (1,1). No padding. Followed by BatchNorm to normalize across channels axis. No activation function applied here.

    Shortcut path components:
    Conv2D with F3 filters of shape (1,1) and stride (s,s). No padding. Followed by BatchNorm to normalize across channels axis.

    Add shortcutpath to the main path.
    Apply ReLU activation function to this.

    """

    # name bases
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # retrieve filters
    F1, F2, F3 = filters

    # save the input value so that it can be added back to the main path (this is the shortcut path)
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)
  
    # Shortcut path
    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Add shortcut value to main path, and pass it through a ReLU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Put everything together to make ResNet with 50 layers
    architecture: INPUT -> ZERO PAD -> STAGE 1: CONV/BATCHNORM/RELU/MAXPOOL -> STAGE 2: CONV_BLOCK/ID_BLOCK x 2 -> STAGE 3: CONV_BLOCK/ID_BLOCK x 2 -> STAGE 4: CONV_BLOCK/ID_BLOCK x 2 -> STAGE 5: CONV_BLOCK/ID_BLOCK x 2 -> AVGPOOL/FLATTEN/FC -> OUTPUT

    Arguments: 
    input_shape -- shape of the images
    classes -- integer, number of classes for fully connected layer with softmax

    Returns: 
    model -- a Keras Model() instance
    """

    # define input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding 3x3
    X = ZeroPadding2D((3,3))(X_input)

    # stage 1. 
    # Conv2D with 64 7x7 filters stride = 2 -> BatchNorm -> MaxPool 3x3 and stride = 2
    X = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), name = 'conv1', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2))(X)

    # stage 2. 
    # conv_block with 3 fiilters, size [64, 64, 256] (the numbers are arbitrary in a way, log2 scale, and just increased 2 fold at each stage ), f = 3, stride = 1. Followed by two identity blocks. 3 sets of filters with 64 filters, 64 filters, 256 filters, f = 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', s = 1)
    X = identity_block(X, f = 3, filters = [64, 64, 256], block = 'b', stage = 2)
    X = identity_block(X, f = 3, filters = [64, 64, 256], block = 'c', stage = 2)

    # stage 3
    # Conv_block with 3 sets of filters of sizes [128, 128, 512], f = 3, s = 2. Followed by three identity blocks with 3 filters of size [128, 128, 512], f = 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], block = 'a', s = 2, stage = 3) 
    X = identity_block(X, f = 3, filters = [128,128,512], block = 'b', stage = 3)
    X = identity_block(X, f = 3, filters = [128,128,512], block = 'c', stage = 3)
    X = identity_block(X, f = 3, filters = [128,128,512], block = 'd', stage = 3)

    # stage 4. 
    # conv_block with 3 sets of filters of sizes [256, 256, 1024], f = 3, s = 2. Followed by 5 identity blocks with 3 sets of filters of sizes 256, 256, 1024, f = 3 
    X = convolutional_block(X, f = 3, filters = [256,256,1024], block = 'a', s = 2, stage = 4) 
    X = identity_block(X, f = 3, filters = [256,256,1024], block = 'b', stage = 4)
    X = identity_block(X, f = 3, filters = [256,256,1024], block = 'c', stage = 4)
    X = identity_block(X, f = 3, filters = [256,256,1024], block = 'd', stage = 4)
    X = identity_block(X, f = 3, filters = [256,256,1024], block = 'e', stage = 4)
    X = identity_block(X, f = 3, filters = [256,256,1024], block = 'f', stage = 4)

    # stage 5
    # conv_block with 3 sets of filters sizes 512, 512, 2048, f = 3, s = 2. Followed by 2 identity blocks with 3 sets of filters of sizes 512, 512, 2048, f =3.
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], block = 'a', s = 2, stage = 5) 
    X = identity_block(X, f = 3, filters = [512, 512, 2048], block = 'b', stage = 5)
    X = identity_block(X, f = 3, filters = [512, 512, 2048], block = 'c', stage = 5)

    # last chunk of layers
    # average pool
    X = AveragePooling2D(pool_size = (2,2), name = 'avg_pool')(X)
    # output layer. Flatten and Fully connected layer
    X = Flatten()(X)
    X = Dense (classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed = 0))(X)

    # create model
    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
    return model

if __name__ == '__main__':
    model = ResNet50(input_shape = (64, 64, 3), classes = 6)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # load data
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # one hot training and test labels
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    print('Number of training examples: '+ str(X_train.shape[0]))
    print('Number of test examples: ' + str(X_test.shape[0]))

    # training
    model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

    # evaluate
    preds = model.evaluate(X_test, Y_test)
    print('Loss: ' + str(preds[0]))
    print('Test accuracy: ' + str(preds[1]))
