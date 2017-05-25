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
    W = tf.Variable(tf.zeros([784, 10]))
    b=tf.Variable(tf.zeros([10]))
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # Model
    Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) +b)

    # Correct answers
    Y_=tf.placeholder(tf.float32, [None, 10])

    # Loss function
    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
 
    # % of correct answers in a batch
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # Optimizer and training step
    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

    # Training loop
    num_steps=2000
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
