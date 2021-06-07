# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:52:02 2018

@author: yeohyeongyu
"""
import tensorflow as tf
import numpy as np

def discriminatorNN_1007(NNinput, isTraining = True, reuse = False, name = 'D_NN1007', device = '/gpu:0'):
    print('Info: Creating NN1007_D:', name)
    with tf.device(device):
        return discriminator(image = NNinput, options = None, reuse = reuse, name = name)

def discriminator(image, options, reuse=False, name='discriminator'):
    DF_DIM = 64
    IMG_CHANNEL = 1
    def first_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
        with tf.variable_scope(name):
            return lrelu(conv2d(input_, out_channels, ks=ks, s=s))
    def conv_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
        with tf.variable_scope(name):
            return lrelu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))
    def last_layer(input_, out_channels, ks=4, s=1, name='disc_conv_layer'):
        with tf.variable_scope(name):
            return tf.contrib.layers.fully_connected(conv2d(input_, out_channels, ks=ks, s=s), out_channels)
        
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        l1 = first_layer(image, DF_DIM, ks=4, s=2, name='disc_layer1')
        l2 = conv_layer(l1, DF_DIM*2, ks=4, s=2, name='disc_layer2')
        l3 = conv_layer(l2, DF_DIM*4, ks=4, s=2, name='disc_layer3')
        l4 = conv_layer(l3, DF_DIM*8, ks=4, s=1, name='disc_layer4')
        l5 = last_layer(l4, IMG_CHANNEL, ks=4, s=1, name='disc_layer5')
        return l5

def generatorNN_1007(NNinput, isTraining = True, reuse = False, name = 'G_NN1007cycle', device = '/gpu:0'):
    print('Info: Creating NN1007_G:', name)
    with tf.device(device):
        return generator(image = NNinput, options = None, reuse = reuse, name = name)

def generator(image, options, reuse=False, name="generator"):
    GF_DIM = 128
    GLF_DIM = 15
    IMG_CHANNEL = 1
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        def conv_layer(input_, out_channels, ks=3, s=1, name='conv_layer'):
            with tf.variable_scope(name):
                return tf.nn.relu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))       
        def gen_module(input_,  out_channels, ks=3, s=1, name='gen_module'):
            with tf.variable_scope(name):
                ml1 = conv_layer(input_, out_channels, ks, s, name=name + '_l1')
                ml2 = conv_layer(ml1, out_channels, ks, s, name=name + '_l2')
                ml3 = conv_layer(ml2, out_channels, ks, s, name=name + '_l3')
                concat_l = input_ + ml3
                m_out = tf.nn.relu(concat_l)
                return m_out
    
        l1 = conv_layer(image, GF_DIM, name='convlayer1') 
        module1 = gen_module(l1, GF_DIM, name='gen_module1')
        module2 = gen_module(module1, GF_DIM, name='gen_module2')
        module3 = gen_module(module2, GF_DIM, name='gen_module3')
        module4 = gen_module(module3, GF_DIM, name='gen_module4')
        module5 = gen_module(module4, GF_DIM, name='gen_module5')
        module6 = gen_module(module5, GF_DIM, name='gen_module6')
        concate_layer = tf.concat([l1, module1, \
                module2, module3, module4, module5, module6], axis=3, name='concat_layer')
        concat_conv_l1 = conv_layer(concate_layer, GF_DIM, ks=3, s=1, name='concat_convlayer1')
        last_conv_layer = conv_layer(concat_conv_l1, GLF_DIM, ks=3, s=1, name='last_conv_layer')
        output= tf.add(conv2d(last_conv_layer, IMG_CHANNEL, ks=3, s=1), image, name = 'output')
        return output 


# network components
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x)

def batchnorm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(input_, axis=3, epsilon=1e-5, \
            momentum=0.1, training=True, \
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def conv2d(batch_input, out_channels, ks=4, s=2, name="cov2d"):
    with tf.variable_scope(name):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=ks, \
            strides=s, padding="valid", \
            kernel_initializer=tf.random_normal_initializer(0, 0.02))

#### loss
def least_square(A, B):
    return tf.reduce_mean((A - B)**2)
    
def cycle_loss(A, F_GA, B, G_FB, lambda_):
    return lambda_ * (tf.reduce_mean(tf.abs(A - F_GA)) + tf.reduce_mean(tf.abs(B - G_FB)))

def identity_loss(A, G_B, B, F_A, gamma):
    return   gamma * (tf.reduce_mean(tf.abs(G_B - B)) + tf.reduce_mean(tf.abs(F_A - A)))
