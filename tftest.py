# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:33:21 2017
TensorFlow tests
@author: mikko hakala
"""

# https://www.tensorflow.org/install/install_windows
# works only for Python 3.5

# Filter out warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Hello world          
import tensorflow as tf
sess = tf.Session()
hello = tf.constant('Hello, TensorFlow!')
print(sess.run(hello))


# Example 1 

# Graph
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session and evaluate tensor c
sess = tf.Session()
print(sess.run(c))

# Close
sess.close()

# Alternative close
with tf.Session() as sess:
  print(sess.run(c))




