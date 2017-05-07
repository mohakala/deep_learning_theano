# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:54:08 2017

XOR gate with lasagne
Follow example in
 https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
@author: Mikko Hakala
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne


def load_dataset():
    pass

def build_mlp(input_var=None):
    # Creates an MLP of two hidden layers of 20 units each
    # ??It applies 20% dropout to the input data and 
    # ??followed by a softmax output layer of 10 units. 
    # ?? 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 4 rows and 2 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    # l_in = lasagne.layers.InputLayer(shape=(None, 1, 4, 2),
    #                                  input_var=input_var)

    l_in = lasagne.layers.InputLayer(shape=(4, 2),
                                     input_var=input_var)


    # No dropout layer
    l_in_drop = l_in
    
    # Add a fully-connected layer of 2 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=2,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

#    # Another 20-unit layer:
#    l_hid2 = lasagne.layers.DenseLayer(
#            l_hid1, num_units=2,
#            nonlinearity=lasagne.nonlinearities.rectify)

    # Finally, we'll add the fully-connected output layer, of 1 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    pass

def main(num_epochs=2): 
    # Get data

    # Inputs and targets
    
    # In MNIST example the data is of the form:
    # (50000, 1, 28, 28) and (50000,)
    
    inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ]
    inputs = np.array(inputs).reshape(-1, 2)

    outputs = [1,0,0,1]
    outputs = np.array(outputs)
    
    print('Data shapes:')
    print(inputs.shape, outputs.shape)

    # Prepare Theano variables for inputs and targets
    #input_var = T.tensor4('inputs')
    #target_var = T.ivector('targets')

    # In AV Theano example:
    input_var = T.matrix('inputs')
    target_var = T.vector('targets')

    # Create network
    network = build_mlp(input_var)

    # Loss expression for training, 
    # i.e., a scalar objective we want to minimize
    # http://lasagne.readthedocs.io/en/stable/modules/objectives.html
    prediction = lasagne.layers.get_output(network)
    #loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)


    # Create a loss expression for validation/testing
    pass

    # Create an expression for the classification accuracy
    pass

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a function to get the predictions
    predicted_fn = theano.function([input_var], prediction)

    # Compile a second function computing the validation loss and accuracy
    pass

    # Launch the training loop
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        
        #for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        #    inputs, targets = batch
        #    train_err += train_fn(inputs, targets)
        #    train_batches += 1

        inputs = inputs
        targets = outputs 
        #print(inputs.shape)
        #print(targets.shape)
        train_err += train_fn(inputs, targets)

        # And a full pass over the validation data:
        pass
        
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #     val_acc / val_batches * 100))

        # print('Predictions:')
        # print(predicted_fn(inputs))
    
    
    # Compute and print test error
    pass

    print('Training done')
    print('  predictions:')
    print(predicted_fn(inputs))


    # TO DO: Make predictions
    

    print('Done')


if __name__ == '__main__':
    main(num_epochs=30000)
