
import keras
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)

def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

# Define our Lrelu
def lrelu(inputs, alpha):
    return keras.layers.LeakyReLU(alpha=alpha).call(inputs)

# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output

def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)







def discriminator_SRGAN(NNinput, isTraining):

    dis_inputs = NNinput

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, isTraining)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:1'):
        with tf.variable_scope('D_unit', reuse = tf.AUTO_REUSE):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'D_block_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'D_block_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'D_block_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'D_block_4')

            # print('after block_4, net.shape: %s' % str(net.shape))

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'D_block_5')

            # print('after block_5, net.shape: %s' % str(net.shape))

            # block 6
            net = discriminator_block(net, 256, 3, 1, 'D_block_6')

            # print('after block_6, net.shape: %s' % str(net.shape))

            # block_7
            net = discriminator_block(net, 256, 3, 2, 'D_block_7')

            # print('after block_7, net.shape: %s' % str(net.shape))

            # The dense layer 1
            with tf.variable_scope('D_dense_layer_1'):
                net = slim.flatten(net)
                print('after dense layer 1 - slim.flatten, net.shape: %s' % str(net.shape))

                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            print('after dense layer 1, net.shape: %s' % str(net.shape))

            # The dense layer 2
            with tf.variable_scope('D_dense_layer_2'):
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

            # print('after dense layer 2, net.shape: %s' % str(net.shape))


    return net

















# Definition of the generator
def generator_SRGAN(gen_inputs, gen_output_channels, reuse=False, isTraining = False):

    resBlockNum = 10


    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, isTraining)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, isTraining)
            net = net + inputs

        return net


    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        print('after stage1_output, net.shape: %s' % str(net.shape))


        # The residual block parts
        for i in range(1, resBlockNum + 1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        print('after residual blocks, net.shape: %s' % str(net.shape))


        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, isTraining)

        net = net + stage1_output

        print('after net + stage1_output, net.shape: %s' % str(net.shape))


        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        print('after subpixelconv_stage1, net.shape: %s' % str(net.shape))


        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        print('after subpixelconv_stage2, net.shape: %s' % str(net.shape))


        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

        print('after output_stage, net.shape: %s' % str(net.shape))


    return net



if __name__ == '__main__':
    pass

    # NNoutput = discriminator_SRGAN(NNinput = tf.placeholder(tf.float32, [None, 512, 512, 1], name = 'low_dose_image'), isTraining = True)
    # generator_SRGAN(gen_inputs = tf.placeholder(tf.float32, [None, 512, 512, 1], name = 'low_dose_image'), gen_output_channels = 1, reuse=False, isTraining = True)