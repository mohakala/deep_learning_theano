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

          
import tensorflow as tf
sess = tf.Session()
hello = tf.constant('Hello, TensorFlow!')
print(sess.run(hello))



