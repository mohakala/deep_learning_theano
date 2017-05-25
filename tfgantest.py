# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:22:13 2017

Study the GAN example from
  http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
  https://github.com/AYLIEN/gan-intro/blob/master/gan.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
import mfunc as m

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


          
class DataDistribution(object):
    # Create samples from Gaussian distribution 
    def __init__(self):
        self.mu = 1
        self.sigma = 0.5

    def sample_gauss(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

    def sample_lin(self, N):
        samples = np.random.rand(N)
        samples += range(N)
        samples = samples/N
        samples.sort()
        return samples

    def sample_sin(self, N):
        # Samples from sin distribution        
        # Random angles
        xrand = np.random.rand(N) * 2 * np.pi
        xrand.sort()
        samples = np.sin(xrand)
        return samples


class GeneratorDistribution(object):
    # Return N input noise samples in the -range ... +range 
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
    # Double amount of nodes. Input: (wx + b) with activation tanh
    h0 = tf.tanh(linear(input, hidden_size * 2, 'd0'))
    h1 = tf.tanh(linear(h0, hidden_size * 2, 'd1'))
    h2 = tf.tanh(linear(h1, hidden_size * 2, 'd2'))
    # Output layer: single probability (sigmoid activation)
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3


def optimizer(loss, var_list):
    # Essentially: 
    #  optimizer=tf.train.GradientDescentOptimizer(0.005)
    #  train_step = optimizer.minimize(cross_entropy)  # == loss
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


def plot_hist(x, y):
    # Continuous plotting of histograms  
    plt.clf()
    plt.hist(x, bins=15, alpha=0.75, label='true')
    plt.hist(y, bins=15, alpha=0.75, label='generated to fool D')
    # plt.xlim(-1, 3)
    plt.legend()

    # Interactive plotting
    plt.pause(0.05)
    # plt.show()    


def main():

    hidden_size=4
    num_steps=10000
    batch_size=10
    
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

    # Loss of generator: Try to get close to true sampling
    loss_g = tf.reduce_mean(-tf.log(D2))

    # Loss of discriminator: Minimize the two errors
    # - pred close to true class=1 value
    # - pred far from forged class=1 value
    loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))


    vars = tf.trainable_variables()
    print('tf.trainable_variables:', vars)
    d_params = [v for v in vars if v.name.startswith('D/')]
    g_params = [v for v in vars if v.name.startswith('G/')]

    # Gradient descent optimizer
    opt_d = optimizer(loss_d, d_params)
    opt_g = optimizer(loss_g, g_params)


    data=DataDistribution()
    gen=GeneratorDistribution(3)
    
    # Auxiliary prints
    print(data.sample_gauss(5))
    print(data.sample_lin(5))
    print(np.random.rand(5))
    print(data.sample_sin(10))

    # Turn on interactive plotting
    plt.ion()

    with tf.Session() as session:
        tf.initialize_all_variables().run()

        print('Loss in G and in D, mean and std of G')

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
            xx = data.sample_gauss(batch_size).reshape(-1,1)
            zz = gen.sample(batch_size).reshape(-1,1)
            session.run([loss_d, opt_d], {
                    x: xx,
                    z: zz
            })
            # Train the generator
            zz = gen.sample(batch_size).reshape(-1,1)
            session.run([loss_g, opt_g], {
                    z: zz
            })
        
            # Print the losses and distributions at some steps
            if(step==2 or step%500==0):
                zz = gen.sample(batch_size).reshape(-1,1)
                forged = session.run(G, {z: zz})
                print(step, session.run(loss_g, {z: zz }),              
                      session.run(loss_d, {x: xx, z: zz }),
                      np.mean(forged), np.std(forged)
                     )
                
                true = xx  # true distribution
                plot_hist(true, forged)
    
 
        # After the models are trained, print some results
        test_batch_size=200

        # Sampling from generator
        zz = gen.sample(test_batch_size)  # batch_size
        preds = session.run(G, {z: np.reshape(zz, (test_batch_size, 1)) })
        preds = sorted(preds)

        print('Type of preds:', type(preds))
        preds=np.asarray(preds)
        print('Type of preds:', type(preds))

        # Print loss for G
        print('Loss for G:')
        print(session.run(loss_g, {z: np.reshape(zz, (test_batch_size, 1)) }))

        # Sampling from true distribution
        true = data.sample_gauss(test_batch_size) 
        print('Type of trues:', type(true))

        # Get network weights
        pass # TODO             


    # Turn off interactive plotting and show the last plot.
    plt.ioff()
    plt.show()

    # Print predicted values
    if(False):
        print('Generated distr', 'True distr')
        for i in range(test_batch_size):
            print(preds[i], true[i])

    print('Generator mean and std:', np.mean(preds), np.std(preds))
    print('True mean and std:', np.mean(true), np.std(true))


    # Plots  
    plt.plot(true,'o', label='sampled true')
    plt.plot(preds,'o', label='generated to fool D to misclassify')
    plt.legend()
    plt.title('True and forged data')
    plt.show()

    # Histograms  
    plt.hist(true, bins=15, alpha=0.75, label='true')
    plt.hist(preds, bins=15, alpha=0.75, label='generated to fool D')
    plt.legend()
    plt.show()    
        
        
        
        
        

        


    
    print('Done')


if __name__ == "__main__":
    main()
