import scipy.io
import scipy.misc
from nst_utils import *
import numpy as np
import tensorflow as tf
import argparse

"""
To build the neural style transfer algorithm:
1. Build content cost function 
2. Build style cost function
3. Combine and calculate cost of the generated pic with
J(G) = alpha*J_content(C,G) + beta*J_style(S,G)

We are trying to match correlations between the activations from a hidden layer l which are from the image being passed through the pre-trained network
"""
# HYPERPARAMETERS
ALPHA = 10
BETA = 40

# ------- functions begin
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost. This is to ensure that the generated image's content matches the original image
    J_content(C,G) = (1/4*nH*nW*nC) * sum_over_all_entries(a^C - a^G)^2

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of image G
    
    Returns:
    J_content -- scalar that is computed with the above equation
    """
   # retrieve dimensions from a_G tensor 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

   # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape = (n_H*n_W, n_C)))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape = (n_H*n_W, n_C)))

   # compute the cost
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))
   
    return J_content

def gram_matrix(A):
    """
    AKA style matrix

    G_i_j = v_i.T * v_j 
    compares how similar v_i is to v_j

    """
    GA = tf.matmul(A, tf.transpose(A))

    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    J_style_layer = (1/(4*nC^2 * (nH * nW)^2))* sum_from_i_to_nC sum_from_j_to_nC (G_S - G_G)^2

    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined by the equation above
    """
    # retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.transpose(
        tf.reshape(a_S, shape = (n_H*n_W, n_C)
        )
    )

    a_G = tf.transpose(
        tf.reshape(a_G, shape = (n_H*n_W, n_C)
        )
    )

    # Compute gram_matrices for images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Compute the loss
    J_style_layer = (1/(4*np.square(n_C)* np.square(np.multiply(n_H, n_W)))) * tf.reduce_sum(
        tf.square(
            tf.subtract(GS,GG)
        )
    )

    return J_style_layer

# so far have only captured the style from only one layer. Next, "merge" style costs from several different layers by adding different weights to each layer's style and computing a linear combination of them
# J_style = sum_over_layers (weight*J_style_layer)
# weights
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes overall style cost from several layers, giving weights to each layer
    
    The style of an image can be represented using the Gram matrix of a hidden layer's activations. Then, this representation is combined from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
    Minimizing the style cost will cause the image G to follow the style of the image S.

    Arguments:
    model -- tensorflow model
    STYLE_LAYERS -- python list containing the names of the layers we want to extract the style from and the coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # select the output tensor of the currently selected layer
        out = model[layer_name]

        # set a_S to be the hidden layer activation from the layer currently selected by running the session on out
        a_S = sess.run(out)

        # set a_G to be the hidden layer activation from the same layer
        # a_G refrences a model[layer_name] that isn't evaluated yet. Later, it will be assigned the image G as the model input, and when the session is run, this will be the activations drawn from the appropriate layer with _G_ as the input
        a_G = out

        # Compute style cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        
        # add coeef * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function given by:
    J(G) = alpha*J_content(C,G) + beta*J_style(S,G)

    Arguments:
    J_content -- from above 
    J_style -- from above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above

    """
    J = alpha * J_content + beta * J_style
    return J

def model_nn(sess, input_image, num_iterations = 200):
    """ Put functions together to make the model"""
    # initialize global variables 
    sess.run(tf.global_variables_initializer())
    # run noisy input image through the model
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        # run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # compute the generated image
        generated_image = sess.run(model['input'])

        # print every 20 iterations:
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print('Iteration ' + str(i) + " :")
            print("total cost = " + str(Jt))
            print('content cost = ' + str(Jc))
            print('style cost = ' + str(Js))
        
        if i%50 == 0:
            # save generated image every 50 iterations
            save_image('output/'+str(i) + '.png', generated_image)

    return generated_image
# ----- functions end

if __name__ == '__main__':
    # CLI 
    parser = argparse.ArgumentParser()
    parser.add_argument('content_image', help = 'path to content image')
    parser.add_argument('style_image', help = 'path to style image')
    parser.add_argument('iterations', type = int, help = 'number of iterations', default = 400)
    args = parser.parse_args()
    ##### put everything together

    # reset the graph
    tf.reset_default_graph()

    # start interactive session
    sess = tf.InteractiveSession()

    # load, reshape, and normalize "content" image
    content_image = scipy.misc.imread(args.content_image)
    content_image = scipy.misc.imresize(content_image, (300,400))
    content_image = reshape_and_normalize_image(content_image)

    # load, reshape, and normalize "style" image
    style_image = scipy.misc.imread(args.style_image)
    style_image = scipy.misc.imresize(style_image, (300,400))
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    # load the VGG19 model
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    # assign the content image to be the input of the VGG model
    sess.run(model['input'].assign(content_image))

    # select the output tensor of layer conv4_2 ...can try tuning by selecting activations from a different layer
    out = model['conv4_2']
    
    # set a_C to be the hidden layer activations from the layer we have selected above (conv4_2)
    a_C = sess.run(out)

    # set a_G to be the hidden layer activation from same layer. Here a_G references model['conv4_2'] but it isn't evaluated yet. Later in the code, the image G will be assigned as the model input, so that when the session is run, this will be the activations drawn from the appropriate layer with G as input
    a_G = out

    # compute content cost
    J_content = compute_content_cost(a_C, a_G)

    # Now the "style" image
    # assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    # compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)

    # now that J_content and J_style are calculated, compute the total cost J
    J = total_cost(J_content, J_style, alpha = ALPHA, beta = BETA)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step to minimize cost J
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image, num_iterations = args.iterations)
