# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:54:08 2017

XOR gate with lasagne
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
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 4, 2),
                                     input_var=input_var)

    # No dropout layer
    l_in_drop = l_in
    
    # Add a fully-connected layer of 20 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=20,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Another 20-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=20,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Finally, we'll add the fully-connected output layer, of 1 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=1,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    pass

def main(): 
    # Get data

    # Inputs and targets
    inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ]
    outputs = [1,0,0,1]


    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')


    # Create network
    network = build_mlp(input_var)


    # Create a loss expression for training
    pass


    # Create update expressions for training
    pass


    # Create a loss expression for validation/testing
    pass
    # Create an expression for the classification accuracy
    pass


    # Compile a function performing a training step on a mini-batch 
    pass


    # Compile a second function computing the validation loss and accuracy
    pass




    # Launch the training loop
    pass


    # Compute and print test error
    pass

    print('Done')


if __name__ == '__main__':
    main()
