# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:22:13 2017

Study the GAN example from
  http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
  https://github.com/AYLIEN/gan-intro/blob/master/gan.py
"""

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
import mfunc as m

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


          
class DataDistribution(object):
    # Return samples from Gaussian distribution 
    def __init__(self):
        self.mu = 2
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    # Return input noise. 
    # Input is stratified (samples uniformly in the range, then add noise)
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


def linear(input, output_dim, scope=None, stddev=1.0):
    # Return y_i = wx_i + b_i
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, hidden_size):
    # One hidden layer, gets as input: softplus(wx + b)
    h0 = tf.nn.softplus(linear(input, hidden_size, 'g0'))
    # Output layer: return the linear value y = wx + b
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, hidden_size):
    # Double amount of nodes. Input: tanh(wx + b)
    h0 = tf.tanh(linear(input, hidden_size * 2, 'd0'))
    h1 = tf.tanh(linear(h0, hidden_size * 2, 'd1'))
    h2 = tf.tanh(linear(h1, hidden_size * 2, 'd2'))
    # Output layer: single probability (sigmoid activation)
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3


def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer



def main():

    hidden_size=4
    num_steps=750
    batch_size=5
    
    print('Number of steps:', num_steps)

    # Make TensorFlow graphs for G and D
    with tf.variable_scope('G'):
        # Input data for generator
        z = tf.placeholder(tf.float32, shape=(None, 1))
        # Create generator nn
        G = generator(z, hidden_size)

    with tf.variable_scope('D') as scope:
        # Input data for discriminator
        x = tf.placeholder(tf.float32, shape=(None, 1))
        # Create discriminator nn
        D1 = discriminator(x, hidden_size)
 
        # Reuse the w and b variables of the discriminator
        scope.reuse_variables()
        # Check the output of discriminator for the forged value from G 
        D2 = discriminator(G, hidden_size)

    # Loss of generator: Try to get close to true value
    loss_g = tf.reduce_mean(-tf.log(D2))

    # Loss of discriminator: Minimize the two errors
    # - pred close to true class=1 value
    # - pred far from forged class=1 value
    loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))


    vars = tf.trainable_variables()
    d_params = [v for v in vars if v.name.startswith('D/')]
    g_params = [v for v in vars if v.name.startswith('G/')]

    # Gradient descent optimizer
    opt_d = optimizer(loss_d, d_params)
    opt_g = optimizer(loss_g, g_params)


    data=DataDistribution()
    gen=GeneratorDistribution(5)


    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for step in range(num_steps):
            # update discriminator
            # Original:
            #x = data.sample(batch_size)
            #z = gen.sample(batch_size)
            #session.run([loss_d, opt_d], {
            #        x: np.reshape(x, (batch_size, 1)),
            #        z: np.reshape(z, (batch_size, 1))
            #})
        
            # Train the discriminator
            # Identical, works, since changed to xx
            xx = data.sample(batch_size).reshape(-1,1)
            zz = gen.sample(batch_size).reshape(-1,1)
            session.run([loss_d, opt_d], {
                    x: xx,
                    z: zz
            })
            # Train the generator
            zz = gen.sample(batch_size)
            session.run([loss_g, opt_g], {
                    z: np.reshape(zz, (batch_size, 1))
            })

 
        # Print some results

        test_batch_size=50

        # Sampling from generator
        zz = gen.sample(test_batch_size)  # batch_size
        preds = session.run(G, {z: np.reshape(zz, (test_batch_size, 1)) })
        preds = sorted(preds)
        print(type(preds))

        # Sampling from true distribution
        true = data.sample(test_batch_size) 

        for i in range(50):
            print(preds[i], true[i])

        print('Generator mean and std:', np.mean(preds), np.std(preds))
        print('True mean and std:', np.mean(true), np.std(true))


        import matplotlib.pyplot as plt
        plt.hist(true, bins='auto')  
        plt.title("Histogram")
        plt.show()
        
        
        # Get network weights
        pass # TODO             

        


    
    print('Done')


if __name__ == "__main__":
    main()
