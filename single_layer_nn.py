'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize
from time import time

# Do not change this


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2 * epsilon) - epsilon
    return W


# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    sigmoid_op = 1.0 / (1.0 + np.exp(-1.0 * z))

    return sigmoid_op


# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0.0

    # Gradient for input weights and output weights
    obj_grad_w1 = 0.0
    obj_grad_w2 = 0.0

    num_samples = training_data.shape[0]

    # Save true outputs
    true_outputs = np.zeros((num_samples, n_class))
    for i in range(num_samples):
        true_outputs[i][training_label[i]] = 1

    # Feedforward propagation

    # Propagating from input layer to hidden layer
    linear_comb_input = np.dot(np.column_stack(
        (training_data, np.ones(num_samples))), w1.T)
    output_hidden = sigmoid(linear_comb_input)
    # Propagating from hidden layer to output layer
    linear_comb_output = np.dot(np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0]))), w2.T)
    output_final = sigmoid(linear_comb_output)
    
    # Finding error and backpropagating

    # Error function
    obj_val = true_outputs * \
        np.log(output_final) + (1 - true_outputs) * np.log(1 - output_final)
    obj_val=(-1/num_samples)*(np.sum(obj_val[:,:]))

    # Find gradients

    # Finding gradient of error between hidden layer and output layer
    delta_output = output_final - true_outputs
    obj_grad_w2 = np.dot(delta_output.T, np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0]))))
    
    # Finding gradient of error between input layer and hidden layer
    obj_grad_w1=np.dot(np.dot(w2.T,delta_output.T)*(((np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0])))).T)*(1-(np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0])))).T)),np.column_stack(
        (training_data, np.ones(num_samples))))
    
    # Remove bias values, as gradient is not calculated for bias
    obj_grad_w1=obj_grad_w1[:-1,:]

    # Calculate regularized error
    obj_val += (lambdaval * (np.sum(np.sum(w1**2)) +
                             np.sum(np.sum(w2**2)))) / (2.0*num_samples)

    # Regularization gradients
    obj_grad_w1 = (obj_grad_w1 + lambdaval * w1) / num_samples
    obj_grad_w2 = (obj_grad_w2 + lambdaval * w2) / num_samples

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate(
        (obj_grad_w1.flatten(), obj_grad_w2.flatten()), 0)

    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    num_samples=data.shape[0]
    # Propagating from input layer to hidden layer
    linear_comb_input = np.dot(np.column_stack(
        (data, np.ones(num_samples))), w1.T)
    output_hidden = sigmoid(linear_comb_input)
    # Propagating from hidden layer to output layer
    linear_comb_output = np.dot(np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0]))), w2.T)
    output_final = sigmoid(linear_comb_output)
    labels = np.argmax(output_final, axis=1)
    return labels

# Do not change this


def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


"""**************Neural Network Script Starts here********************************"""
start_time=time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate(
    (initial_w1.flatten(), initial_w2.flatten()), 0)
# set the regularization hyper-parameter
lambdaval = 10
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True,
                     args=args, method='CG', options=opts)
params = nn_params.get('x')
# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data)
# find the accuracy on Training Dataset
print('\nTraining set Accuracy:' +
      str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, validation_data)
# find the accuracy on Validation Dataset
print('\nValidation set Accuracy:' + str(100 *
                                          np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset
print('\nTest set Accuracy:' +
      str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
end_time = time()
print('\nTime Taken:' + str(end_time-start_time)+'s')