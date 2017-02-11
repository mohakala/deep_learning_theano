# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:15:04 2017

https://lasagne.readthedocs.io/en/latest/user/tutorial.html
https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

Structure of the code but not code itself
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
    pass


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
    pass


def build_cnn(input_var=None):
    pass


def main(model='mlp', num_epochs=500):
    # Load the dataset
    # Prepare Theano variables for inputs and targets
    # Create neural network model (depending on first command line parameter)
    # Create a loss expression for training, i.e., a scalar objective we want
    #   to minimize (for our multi-class problem, it is the cross-entropy loss):
    # Create update expressions for training, i.e., how to modify the
    #   parameters at each training step. 
    # Create a loss expression for validation/testing. 
    # Also create an expression for the classification accuracy
    # Compile a function performing a training step on a mini-batch (by giving
    #   the updates dictionary) and returning the corresponding training loss
    # Compile a second function computing the validation loss and accuracy:
    # Launch the training loop:
    # - for epoch in range(num_epochs):
    #   - for batch in iterate_minibatches(): 
    #   x full pass over the training data 
    #   x full pass over the validation data
    # Compute and print the test error
    #   - for batch in iterate_minibatches(): 
    # Optionally, dump the network weights to a file

    print('Done')
    #assert False, "-- FORCED STOP --"


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)


