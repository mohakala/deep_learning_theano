# -*- coding: utf-8 -*-
"""
https://www.tensorflow.org/get_started/tflearn
with some modifications
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib


import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Use ERROR to not report deprecation warnings
# Use INFO to report info during training
tf.logging.set_verbosity(tf.logging.ERROR)


# Data sets
IRIS_TRAINING = "../datasets/iris_training.csv"
#IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "../datasets/iris_test.csv"
#IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]


  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="/tmp/iris_model")


  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  #assert False, "x"

  # Fit model
  classifier.fit(input_fn=get_train_inputs, steps=10)

  #assert False, "x"


  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

  # predictions = list(classifier.predict(input_fn=new_samples))
  predictions = list(classifier.predict_classes(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))


if __name__ == "__main__":
    main()
