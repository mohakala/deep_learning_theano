# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:23:38 2017

MNIST example following
  https://www.youtube.com/watch?v=vq2nnJ4g6N0&t=2460s
@author: mikko hakala
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def main():
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../datasets/MNIST_data/", one_hot=True)                
            
    # Placeholders and variables
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    X = tf.reshape(X, [-1, 784])
    # alternatively, give directly the shape:
    #   X = tf.placeholder(tf.float32, [None, 784])
    # correct answers
    Y_=tf.placeholder(tf.float32, [None, 10])



    ## A. Single-layer perceptron (slp) 
    make_slp = False
    if(make_slp):    
        W = tf.Variable(tf.zeros([784, 10]))
        b=tf.Variable(tf.zeros([10]))
        # Model
        Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)


    ## B. Multilayer perceptron
    make_mlp = True
    if(make_mlp):
        # Variables
        A = 784
        K = 200
        L = 100
        M = 60
        N = 30
        O = 10
        W1 = tf.Variable(tf.truncated_normal([A, K], stddev=0.1))
        B1 = tf.Variable(tf.zeros([K]))
        W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
        B2 = tf.Variable(tf.zeros([L]))
        W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        B3 = tf.Variable(tf.zeros([M]))
        W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        B4 = tf.Variable(tf.zeros([N]))
        W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
        B5 = tf.Variable(tf.zeros([O]))
        # Model
        Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
        Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)


    # Initialize
    init = tf.global_variables_initializer()
    # init = tf.initialize_all_variables()  # gives warning

    
    # Loss function
    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
 
    # % of correct answers in a batch
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # Optimizer and training step
    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

    # Training loop
    num_steps=3000
    sess = tf.Session()
    sess.run(init)
    for i in range(num_steps):
        # Load training data in batches
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data={X: batch_X, Y_: batch_Y}
        
        # Train
        sess.run(train_step, feed_dict=train_data)
        
        # Print accuracy and cross-entropy
        if(i%100==0):
            acc, succ = sess.run([accuracy, cross_entropy], feed_dict=train_data)
            test_data = {X: mnist.test.images, Y_: mnist.test.labels}
            acc_test, succ_test = sess.run([accuracy, cross_entropy], feed_dict=test_data)
            print('i, Acc, Cross-Ent:', i, acc, succ, '\tTest data:', acc_test, succ_test)



    print('Done')


if __name__ == "__main__":
    main()
