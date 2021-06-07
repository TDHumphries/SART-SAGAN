import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# from utils import *
# from conv_helper import *


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False):
    with tf.variable_scope(scope_name, reuse = reuse):
        filter = tf.Variable(tf.random_normal([ksize, ksize, in_channels, out_channels], stddev=0.03))
        output = tf.nn.conv2d(input_image, filter, strides=[1, stride, stride, 1], padding='SAME')
        output = slim.batch_norm(output)
        if activation_function:
            output = activation_function(output)
        return output, filter

# def conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False):
#     with tf.variable_scope(scope_name, reuse = reuse):
#         filter = tf.Variable(tf.random_normal([ksize, ksize, in_channels, out_channels], stddev=0.03))
#         filter = tf.Variable(tf.random_normal([1,1,1,1]))
#         output = tf.nn.conv2d(input_image, filter, strides=[1, stride, stride, 1], padding='SAME')
#         # output = slim.batch_norm(output)
#         # if activation_function:
#         #     output = activation_function(output)
#         return output, filter

def residual_layer(input_image, ksize, in_channels, out_channels, stride, scope_name):
    with tf.variable_scope(scope_name):
        output, filter = conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name+"_conv1")
        output, filter = conv_layer(output, ksize, out_channels, out_channels, stride, scope_name+"_conv2")
        output = tf.add(output, tf.identity(input_image))
        return output, filter

def transpose_deconvolution_layer(input_tensor, new_shape, scope_name, stride = 3):
    with tf.variable_scope(scope_name):
        W = tf.get_variable(scope_name + '_W', shape=[3, 3, input_tensor.shape[3], new_shape[3]], initializer=tf.truncated_normal_initializer(0., 0.005))
        output = tf.nn.conv2d_transpose(input_tensor, W, output_shape=new_shape, strides=[1, stride, stride, 1], padding='SAME')
        # deconv = tf.nn.conv2d(input_tensor, W, [1,stride,stride,1], padding='SAME')
        output = tf.nn.relu(output)
        return output

def resize_deconvolution_layer(input_tensor, new_shape, scope_name):
    with tf.variable_scope(scope_name):
        output = tf.image.resize_images(input_tensor, (new_shape[1], new_shape[2]), method=1)
        output, unused_weights = conv_layer(output, 3, int(input_tensor.shape[3]), new_shape[3], 1, scope_name+"_deconv")
        return output

def deconvolution_layer(input_tensor, new_shape, scope_name):
    return resize_deconvolution_layer(input_tensor, new_shape, scope_name)

def output_between_zero_and_one(output):
    output +=1
    return output/2

'''============================================================================================================================'''
'''============================================================================================================================'''
'''============================================================================================================================'''
'''============================================================================================================================'''
'''============================================================================================================================'''
# def generator_(input):
#     data = tf.layers.conv2d(input, 1, 3, padding = 'same', activation = tf.nn.relu, name = 'G_g_conv1')
#     return data

# def generator_oriIDGAN(input):
#     print(input.shape)
#     conv1, conv1_weights = conv_layer(input, 9, 1, 16, 1, "G_conv1")
#     print(conv1.shape)
#     conv2, conv2_weights = conv_layer(conv1, 3, 16, 32, 1, "G_conv2")
#     print(conv2.shape)
#     conv3, conv3_weights = conv_layer(conv2, 3, 32, 64, 1, "G_conv3")
#     print(conv3.shape)

#     res1, res1_weights = residual_layer(conv3, 3, 64, 64, 1, "G_res1")
#     print(res1.shape)
#     res2, res2_weights = residual_layer(res1, 3, 64, 64, 1, "G_res2")
#     print(res2.shape)
#     res3, res3_weights = residual_layer(res2, 3, 64, 64, 1, "G_res3")
#     print(res3.shape)

#     deconv1 = deconvolution_layer(res3, [input.shape[0], 256, 256, 32], 'G_deconv1')
#     print(deconv1.shape)
#     deconv2 = deconvolution_layer(deconv1, [input.shape[0], 512, 512, 16], "G_deconv2")
#     print(deconv2.shape)

#     deconv2 = deconv2 + conv1
#     print(deconv2.shape)

#     conv4, conv4_weights = conv_layer(deconv2, 9, 16, 1, 1, "G_conv5", activation_function=tf.nn.tanh)
#     print(conv4.shape)

#     conv4 = conv4 + input
#     print(conv4.shape)
#     output = output_between_zero_and_one(conv4)
#     print(output.shape)

#     return output


def generator(input):
    data = input
    # conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False)
    print(data.shape)

    conv1, _ = conv_layer(data, 9, 1, 16, 1, "G_conv1")
    print(conv1.shape)
    conv2, _ = conv_layer(conv1, 3, 16, 32, 1, "G_conv2")
    print(conv2.shape)
    conv3, _ = conv_layer(conv2, 3, 32, 48, 1, "G_conv3")
    print(conv3.shape)
    conv4, _ = conv_layer(conv3, 3, 48, 64, 1, "G_conv4")
    print(conv4.shape)

    data = conv4

    # residual_layer(input_image, ksize, in_channels, out_channels, stride, scope_name)
    data, _ = residual_layer(data, 3, 64, 64, 1, "G_res1")
    print(data.shape)
    data, _ = residual_layer(data, 3, 64, 64, 1, "G_res2")
    print(data.shape)
    data, _ = residual_layer(data, 3, 64, 64, 1, "G_res3")
    print(data.shape)

    # deconvolution_layer(input_tensor, new_shape, scope_name)
    deconv3 = deconvolution_layer(data, [input.shape[0], 256, 256, 48], 'G_deconv3')
    print(deconv3.shape)
    deconv2 = deconvolution_layer(deconv3, [input.shape[0], 512, 512, 32], "G_deconv2")
    print(deconv2.shape)
    deconv1 = deconvolution_layer(deconv2, [input.shape[0], 512, 512, 16], "G_deconv1")
    print(deconv1.shape)
    deconv0 = deconvolution_layer(deconv1, [input.shape[0], 512, 512, 1], "G_deconv0")
    print(deconv0.shape)

    # # conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False)
    # conv4, conv4_weights = conv_layer(deconv0, 9, 16, 1, 1, "G_conv5", activation_function=tf.nn.tanh)
    # print(conv4.shape)

    # conv4 = conv4 + input
    # print(conv4.shape)
    output = output_between_zero_and_one(deconv0)
    print(output.shape)

    return output


def discriminator_(input, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('D_d_block1', reuse = reuse):
        data = tf.layers.conv2d(input, 1, 3, padding = 'same', activation = tf.nn.relu, name = 'd_conv1')
    return data
def discriminator(input, reuse=tf.AUTO_REUSE):
    print(input.shape)
    conv1, conv1_weights = conv_layer(input, 4, 1, 24, 2, "D_conv1", reuse=reuse)
    print(conv1.shape)
    conv2, conv2_weights = conv_layer(conv1, 4, 24, 48, 4, "D_conv2", reuse=reuse)
    print(conv2.shape)
    conv3, conv3_weights = conv_layer(conv2, 4, 48, 64, 4, "D_conv3", reuse=reuse)
    print(conv3.shape)
    conv4, conv4_weights = conv_layer(conv3, 4, 64, 32, 4, "D_conv4", reuse=reuse)
    print(conv4.shape)
    conv5, conv5_weights = conv_layer(conv4, 4, 32, 1, 4, "D_conv5", activation_function=tf.nn.sigmoid, reuse=reuse)
    print(conv5.shape)

    return conv5


'''============================================================================================================================'''
'''============================================================================================================================'''
'''============================================================================================================================'''
'''============================================================================================================================'''
'''============================================================================================================================'''


# import vgg16
# def get_style_layer_vgg16(image):
#     net = vgg16.get_vgg_model()
#     style_layer = 'conv2_2/conv2_2:0'
#     feature_transformed_image = tf.import_graph_def(
#         net['graph_def'],
#         name='vgg',
#         input_map={'images:0': image},return_elements=[style_layer])
#     feature_transformed_image = (feature_transformed_image[0])
#     return feature_transformed_image


def get_pixel_loss(target,prediction):
    pixel_difference = target - prediction
    pixel_loss = tf.nn.l2_loss(pixel_difference)
    return pixel_loss

def get_style_loss(target,prediction):
    return 0.0
    # feature_transformed_target = get_style_layer_vgg16(target)
    # feature_transformed_prediction = get_style_layer_vgg16(prediction)
    # feature_count = tf.shape(feature_transformed_target)[3]
    # style_loss = tf.reduce_sum(tf.square(feature_transformed_target-feature_transformed_prediction))
    # style_loss = style_loss/tf.cast(feature_count, tf.float32)
    # return style_loss


def get_smooth_loss(image):
    batch_count = tf.shape(image)[0]
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    horizontal_normal = tf.slice(image, [0, 0, 0,0], [batch_count, image_height, image_width-1,1])
    horizontal_one_right = tf.slice(image, [0, 0, 1,0], [batch_count, image_height, image_width-1,1])
    vertical_normal = tf.slice(image, [0, 0, 0,0], [batch_count, image_height-1, image_width,1])
    vertical_one_right = tf.slice(image, [0, 1, 0,0], [batch_count, image_height-1, image_width,1])
    smooth_loss = tf.nn.l2_loss(horizontal_normal-horizontal_one_right)+tf.nn.l2_loss(vertical_normal-vertical_one_right)
    return smooth_loss