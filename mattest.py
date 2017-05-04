# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:54:08 2017

Test of NN
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
    l_out=0
    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    pass

def main(): 
    # Get data
    pass

    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')


    # Create network
    input_var = 0
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