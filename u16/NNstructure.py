

import argparse
import copy
import glob
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

import layerBlocks
import patches
import NNstructure_SAGAN
import SAGAN_ops


def generatorNN_simpleGAN(input, is_training = True, output_channels = 1):
	data = input
	with tf.variable_scope('G_block1', reuse = tf.AUTO_REUSE):
		data = tf.layers.conv2d(data, 64, 3, padding = 'same', activation = tf.nn.relu, name = 'G_conv1')

	for layers in range(2, 16 + 1):
		with tf.variable_scope('G_block%d' % layers, reuse = tf.AUTO_REUSE):
			data = tf.layers.conv2d(data, 64, 3, padding = 'same', name = 'G_conv%d' % layers, use_bias = False)
			data = tf.nn.relu(tf.layers.batch_normalization(data, training = is_training))
	with tf.variable_scope('G_block17', reuse = tf.AUTO_REUSE):
		data = tf.layers.conv2d(data, output_channels, 3, padding = 'same', name = 'G_conv17')
	# return output
	return input - data


def discriminatorNN_simpleGAN(NNinput, isTraining, reuse):
	with tf.device(patches.patch_GPUdistribution_for_NNstructure_discriminatorNN_simpleGAN()):
		with tf.variable_scope('D_block', reuse = tf.AUTO_REUSE):
			data = NNinput
			data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv1', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv2', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv3', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv4', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv5', reuse = tf.AUTO_REUSE)
	return data


'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''



def generatorNN_modifiedDnCNN(NNinput, isTraining = True, output_channels = 1, name = 'G_block', device = '/gpu:0'):
	data = NNinput
	with tf.device(device):
		with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
			data = tf.layers.conv2d(data, 64, 3, padding = 'same', activation = tf.nn.relu, name = 'G_conv1')
			for layers in range(2, 16 + 1):
				data = tf.layers.conv2d(data, 64, 3, padding = 'same', name = 'G_conv%d' % layers, use_bias = False)
				data = tf.nn.relu(tf.layers.batch_normalization(data, training = isTraining))
			data = tf.layers.conv2d(data, output_channels, 3, padding = 'same', name = 'G_conv17')
	# if name == 'G_Gnet':
	# 	return NNinput - data
	# elif name == 'G_Fnet':
	# 	return data
	# else:
		# pass
	return NNinput - data

def discriminatorNN_changedSimpleGAN(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'D_block', device = '/gpu:1'):
	with tf.device(device):
		with tf.variable_scope(name, reuse = tf.AUTO_REUSE):

			data = NNinput
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv1', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv2', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv3', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv4', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv5', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv6', reuse = tf.AUTO_REUSE)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv7', reuse = tf.AUTO_REUSE)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv8', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv9', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv10', reuse = reuse)
			data = slim.flatten(data)
			data = layerBlocks.lrelu(data, 0.2)
			data = layerBlocks.denselayer(data, 1)
			data = tf.nn.sigmoid(data)

	return data



'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''


def generatorNN_cycleGAN(image, reuse = False, name = "G_"):

	gf_dim = 128
	glf_dim = 15
	img_channel = 1

	def lrelu(x, leak=0.2, name="lrelu"):
		with tf.variable_scope(name):
			return tf.maximum(x, leak*x)

	def batchnorm(input_, name="batch_norm"):
		with tf.variable_scope(name):
			return tf.layers.batch_normalization(input_, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

	def conv2d(batch_input, out_channels, ks=4, s=2, name="cov2d"):
		with tf.variable_scope(name):
			padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
			return tf.layers.conv2d(padded_input, out_channels, kernel_size=ks, strides=s, padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


	with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
		# if reuse:
		# 	tf.get_variable_scope().reuse_variables()
		# else:
		# 	assert tf.get_variable_scope().reuse is False

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

		l1 = conv_layer(image, gf_dim, name='convlayer1') 
		module1 = gen_module(l1, gf_dim, name='gen_module1')
		module2 = gen_module(module1, gf_dim, name='gen_module2')
		module3 = gen_module(module2, gf_dim, name='gen_module3')
		module4 = gen_module(module3, gf_dim, name='gen_module4')
		module5 = gen_module(module4, gf_dim, name='gen_module5')
		module6 = gen_module(module5, gf_dim, name='gen_module6')
		concate_layer = tf.concat([l1, module1, module2, module3, module4, module5, module6], axis=3, name='concat_layer')
		concat_conv_l1 = conv_layer(concate_layer, gf_dim, ks=3, s=1, name='concat_convlayer1')
		last_conv_layer = conv_layer(concat_conv_l1, glf_dim, ks=3, s=1, name='last_conv_layer')
		output= tf.add(conv2d(last_conv_layer, img_channel, ks=3, s=1), image, name = 'output')
		return output 



def discriminatorNN_cycleGAN(image, reuse = tf.AUTO_REUSE, name = 'D_', device = '/gpu:1'):

	df_dim = 64
	img_channel = 1

	def lrelu(x, leak=0.2, name="lrelu"):
		with tf.variable_scope(name):
			return tf.maximum(x, leak*x)
			
	def batchnorm(input_, name="batch_norm"):
		with tf.variable_scope(name):
			return tf.layers.batch_normalization(input_, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

	def conv2d(batch_input, out_channels, ks=4, s=2, name="cov2d"):
		with tf.variable_scope(name):
			padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
			return tf.layers.conv2d(padded_input, out_channels, kernel_size=ks, strides=s, padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))

	def first_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
		with tf.variable_scope(name):
			return lrelu(conv2d(input_, out_channels, ks=ks, s=s))
	def conv_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
		with tf.variable_scope(name):
			return lrelu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))
	def last_layer(input_, out_channels, ks=4, s=1, name='disc_conv_layer'):
		with tf.variable_scope(name):
			return tf.layers.dense(conv2d(input_, out_channels, ks=ks, s=s), out_channels)
			# return tf.contrib.layers.fully_connected(conv2d(input_, out_channels, ks=ks, s=s), out_channels)

	with tf.device(device):
		with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
			# if reuse:
			# 	tf.get_variable_scope().reuse_variables()
			# else:
			# 	assert tf.get_variable_scope().reuse is False
			l1 = first_layer(image, df_dim, ks=4, s=2, name='disc_layer1')
			l2 = conv_layer(l1, df_dim*2, ks=4, s=2, name='disc_layer2')
			l3 = conv_layer(l2, df_dim*4, ks=4, s=2, name='disc_layer3')
			l4 = conv_layer(l3, df_dim*8, ks=4, s=1, name='disc_layer4')
			# l5 = last_layer(l4, img_channel, ks=4, s=1, name='disc_layer5')
			return l4





'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''


def generatorNN_GTest(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = '/gpu:0'):
	print('Info: NNstructure.generatorNN_GTest() start')
	with tf.variable_scope(name, reuse = reuse):
		data, E0, E1, E2, E3 = NNstructure_SAGAN.generatorNN(NNinput = NNinput, is_training = isTraining, reuse = reuse, device = device)
		# data = NNinput
	return data

def generatorNN_DDGAN(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = '/gpu:0'):
	print('Info: NNstructure.generatorNN_DDGAN() start')
	with tf.variable_scope(name, reuse = reuse):
		return NNstructure_SAGAN.generatorNN(NNinput = NNinput, is_training = isTraining, reuse = reuse, device = device)


def discriminatorNN_DTest(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'D_testDiscriminator', device = '/gpu:1'):
	print('Info: NNstructure.discriminatorNN_DTest() start.')
	with tf.device(device):
		with tf.variable_scope(name, reuse = reuse):
			print('---------------------------D start-----------------------------')

			print('\tD_NNinput:', NNinput.shape)

			data = NNinput
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv1', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv2', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv3', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv4', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv5', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv6', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv7', reuse = tf.AUTO_REUSE)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv8', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv9', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv10', reuse = reuse)
			data = slim.flatten(data)
			print('\tD_flattenLayer:', data.shape)
			data = layerBlocks.lrelu(data, 0.2)
			data = layerBlocks.denselayer(data, 1)
			data = tf.nn.sigmoid(data)
			print('\tD_output:', data.shape)
			print('------------------------------D finish-----------------------------')

	return data


def discriminatorNN_DDGAN_L(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'L_DDGAN', device = '/gpu:1'):
	print('Info: NNstructure.discriminatorNN_DDGAN_L start.')
	with tf.device(device):
		with tf.variable_scope(name, reuse = reuse):
			print('---------------------------L start-----------------------------')

			print('\tL_NNinput:', NNinput.shape)

			data = NNinput
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'L_conv1', reuse = tf.AUTO_REUSE)
			print('\tL_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'L_conv2', reuse = tf.AUTO_REUSE)
			print('\tL_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 4, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'L_conv3', reuse = tf.AUTO_REUSE)
			print('\tL_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 3, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'L_conv4', reuse = tf.AUTO_REUSE)
			print('\tL_convLayer:', data.shape)
			# data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv5', reuse = tf.AUTO_REUSE)
			# print('\tL_convLayer:', data.shape)
			# data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv6', reuse = tf.AUTO_REUSE)
			# print('\tL_convLayer:', data.shape)
			# data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv7', reuse = tf.AUTO_REUSE)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv8', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv9', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv10', reuse = reuse)
			data = slim.flatten(data)
			print('\tL_flattenLayer:', data.shape)
			data = layerBlocks.lrelu(data, 0.2)
			data = layerBlocks.denselayer(data, 1)
			data = tf.nn.sigmoid(data)
			print('\tL_output:', data.shape)
			print('------------------------------L finish-----------------------------')

	return data








'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''
'''=================================================================================================================================='''






def generatorNN_716(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = ['/gpu:0']):
	print('Info: NNstructure.generatorNN_716() start')
	with tf.variable_scope(name, reuse = reuse):
		print('\tNNinput =', NNinput.shape)
		x = NNinput
		encoderLayerNum = 10
		decoderLayerNum = 10
		channelGrowLayerNum = 5
		ch = 2
		layerNum = 20
		innerLayerNum = 3

		print('-------------------------G start----------------------------')
		with tf.device(device[0]):
			for i in range(encoderLayerNum):
				x = tf.layers.conv2d(x, ch, 5, strides = (1, 1), padding = 'valid', activation = tf.nn.relu, name = 'G_encoConv_'+str(i), reuse = tf.AUTO_REUSE)
				# x = batch_norm(x, is_training, scope='enco_batch_norm_' + str(i+1))
				x = tf.nn.relu(x)
				print('\tG_encoLayer:', x.shape)
				ch = ch * 2 if i < channelGrowLayerNum else ch
		E0x = x
		print('------------------------------------------------------------')
		
		with tf.device(device[0]):
			# Self Attention
			y = x
			y = tf.layers.conv2d(y, 16, 4, strides = (4, 4), padding = 'valid', activation = tf.nn.relu, name = 'G_innerEnco_'+str(i), reuse = tf.AUTO_REUSE)
			print('\tG_inner-encoLayer:', y.shape)
			E1x = y

			y = NNstructure_SAGAN.attentionBlock(y, 16, sn=True, scope="attention", reuse=reuse)
			print('\ty after attentionBlock:', y.shape)
			E2x = y

			y = tf.layers.conv2d_transpose(inputs=y, filters=64, kernel_size=4, strides=(4, 4), padding='valid', use_bias=True)
			print('\tG_inner-decoLayer:', y.shape)
			E3x = y
			x = tf.nn.relu(x) + tf.nn.relu(y)
			print('\tG_mergeLayer:', x.shape)

		E4x = x
		print('------------------------------------------------------------')
		with tf.device(device[0]):
			zz = list(range(decoderLayerNum - 1))
			zz.reverse()
			for i in zz:
				x = SAGAN_ops.deconv(x, channels=ch, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='deconv_' + str(i))
				# x = batch_norm(x, is_training, scope='deco_batch_norm_' + str(i))
				x = tf.nn.relu(x)
				print('\tG_decoLayer:', x.shape)
				ch = ch // 2 if i < channelGrowLayerNum else ch

			x = SAGAN_ops.deconv(x, channels=1, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='G_deconv_logit')
			print('\tG_deconv-Final:', x.shape)
			x = tf.tanh(x)
			print('\tG_tanh:', x.shape)
		print('---------------------------G end----------------------------')

		return x, E0x, E1x, E2x, E3x, E4x




def discriminatorNN_716(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'D_testDiscriminator', device = '/gpu:1'):
	print('Info: NNstructure.discriminatorNN_716() start.')
	with tf.device(device):
		with tf.variable_scope(name, reuse = reuse):
			print('---------------------------D start-----------------------------')

			print('\tD_NNinput:', NNinput.shape)

			data = NNinput
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv1', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv2', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv3', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv4', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv5', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv6', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv7', reuse = tf.AUTO_REUSE)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv8', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv9', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv10', reuse = reuse)
			data = slim.flatten(data)
			print('\tD_flattenLayer:', data.shape)
			data = layerBlocks.lrelu(data, 0.2)
			data = layerBlocks.denselayer(data, 1)
			data = tf.nn.sigmoid(data)
			print('\tD_output:', data.shape)
			print('------------------------------D finish-----------------------------')

	return data










def generatorNN_721(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = ['/gpu:0'], verbose = True):
	# only can train on kaczmarz GPU x2
	print('Info: NNstructure.generatorNN_721() start')
	if not verbose: print('Info: creating NN721 generator...')
	with tf.variable_scope(name, reuse = reuse):
		if verbose: print('\tNNinput =', NNinput.shape)
		x = NNinput
		encoderLayerNum = 10
		decoderLayerNum = 10
		channelGrowLayerNum = 5
		ch = 2
		layerNum = 20
		innerLayerNum = 3

		if verbose: print('-------------------------G start----------------------------')
		with tf.device(device[0]):
			for i in range(encoderLayerNum):
				x = tf.layers.conv2d(x, ch, 5, strides = (1, 1), padding = 'valid', activation = tf.nn.relu, name = 'G_encoConv_'+str(i), reuse = tf.AUTO_REUSE)
				# x = batch_norm(x, is_training, scope='enco_batch_norm_' + str(i+1))
				x = tf.nn.relu(x)
				print('\tG_encoLayer:', x.shape)
				ch = ch * 2 if i < channelGrowLayerNum else ch
			E0x = x
			if verbose: print('------------------------------------------------------------')
		
		with tf.device(device[-1]):
			# Self Attention
			y = x
			y = tf.layers.average_pooling2d(inputs = y, pool_size = (3, 3), strides = (3, 3), padding = 'valid', data_format = 'channels_last', name = 'G_innerEnco_'+str(i))
			if verbose: print('\tG_inner-encoLayer:', y.shape)
			E1x = y

			y, attMap = NNstructure_SAGAN.attentionBlock2(y, 64, sn=True, scope="attention", reuse=reuse)
			if verbose: print('\ty after attentionBlock:', y.shape)
			E2x = y
		
			# y = tf.layers.conv2d_transpose(inputs=y, filters=64, kernel_size=4, strides=(4, 4), padding='valid', use_bias=True)
			y = tf.image.resize_images(images = y, size = (y.shape[1]*3+1, y.shape[2]*3+1), method = 1)
			if verbose: print('\tG_inner-decoLayer:', y.shape)
			E3x = y

			Wx = tf.get_variable('xWeight', shape=[x.shape[3]], dtype=tf.float32, trainable=True)
			Wy = tf.get_variable('yWeight', shape=[y.shape[3]], dtype=tf.float32, trainable=True)
			x = tf.nn.relu(x * Wx) + tf.nn.relu(y * Wy)
			if verbose: print('\tG_mergeLayer:', x.shape)

			E4x = x
			if verbose: print('------------------------------------------------------------')
		with tf.device(device[0]):	
			zz = list(range(decoderLayerNum - 1))
			zz.reverse()
			for i in zz:
				x = SAGAN_ops.deconv(x, channels=ch, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='deconv_' + str(i))
				# x = batch_norm(x, is_training, scope='deco_batch_norm_' + str(i))
				x = tf.nn.relu(x)
				if verbose: print('\tG_decoLayer:', x.shape)
				ch = ch // 2 if i < channelGrowLayerNum else ch

			x = SAGAN_ops.deconv(x, channels=1, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='G_deconv_logit')
			if verbose: print('\tG_deconv-Final:', x.shape)
			x = tf.tanh(x)
			if verbose: print('\tG_tanh:', x.shape)
		if verbose: print('---------------------------G end----------------------------')

		return x, attMap, E0x, E1x, E2x, E3x, E4x, Wx, Wy

'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''

def generatorNN_721NSA(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = ['/gpu:0']):
	print('Info: NNstructure.generatorNN_721NSA() start')
	with tf.variable_scope(name, reuse = reuse):
		print('\tNNinput =', NNinput.shape)
		x = NNinput
		encoderLayerNum = 10
		decoderLayerNum = 10
		channelGrowLayerNum = 5
		ch = 2
		layerNum = 20
		innerLayerNum = 3

		print('-------------------------G start----------------------------')
		with tf.device(device[0]):
			for i in range(encoderLayerNum):
				x = tf.layers.conv2d(x, ch, 5, strides = (1, 1), padding = 'valid', activation = tf.nn.relu, name = 'G_encoConv_'+str(i), reuse = tf.AUTO_REUSE)
				# x = batch_norm(x, is_training, scope='enco_batch_norm_' + str(i+1))
				x = tf.nn.relu(x)
				print('\tG_encoLayer:', x.shape)
				ch = ch * 2 if i < channelGrowLayerNum else ch
		E0x = x
		print('------------------------------------------------------------')
		E4x = x
		print('------------------------------------------------------------')
		with tf.device(device[0]):
			zz = list(range(decoderLayerNum - 1))
			zz.reverse()
			for i in zz:
				x = SAGAN_ops.deconv(x, channels=ch, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='deconv_' + str(i))
				# x = batch_norm(x, is_training, scope='deco_batch_norm_' + str(i))
				x = tf.nn.relu(x)
				print('\tG_decoLayer:', x.shape)
				ch = ch // 2 if i < channelGrowLayerNum else ch

			x = SAGAN_ops.deconv(x, channels=1, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='G_deconv_logit')
			print('\tG_deconv-Final:', x.shape)
			x = tf.tanh(x)
			print('\tG_tanh:', x.shape)
		print('---------------------------G end----------------------------')

		return x, None, E0x, None, None, None, E4x, None, None


'''=============================================================================================================='''
'''=============================================================================================================='''

def discriminatorNN_721NSA(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'D_testDiscriminator', device = '/gpu:1'):
	print('Info: NNstructure.discriminatorNN_21NSA() start.')
	with tf.device(device):
		with tf.variable_scope(name, reuse = reuse):
			print('---------------------------D start-----------------------------')

			print('\tD_NNinput:', NNinput.shape)

			data = NNinput
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv1', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv2', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv3', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv4', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv5', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			data = tf.layers.conv2d(data, 1, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv6', reuse = tf.AUTO_REUSE)
			print('\tD_convLayer:', data.shape)
			# data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv7', reuse = tf.AUTO_REUSE)
			# print('\tD_convLayer:', data.shape)
			# data = tf.layers.conv2d(data, 1, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv8', reuse = reuse)
			# print('\tD_convLayer:', data.shape)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv9', reuse = reuse)
			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv10', reuse = reuse)
			data = slim.flatten(data)
			print('\tD_flattenLayer:', data.shape)
			data = layerBlocks.lrelu(data, 0.2)
			# data = layerBlocks.denselayer(data, 1)
			data = tf.nn.sigmoid(data)
			# data = tf.squeeze(data)
			data = tf.reduce_sum(data)
			print('\tD_output:', data.shape)
			print('------------------------------D finish-----------------------------')

	return data




'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''


# def generatorNN_728(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = ['/gpu:0']):
# 	print('Info: NNstructure.generatorNN_716() start')
# 	# with tf.variable_scope(name = '728generator', reuse = reuse):
# 	# 	print('\tNNinput =', NNinput.shape)
# 	# 	x = NNinput
# 	# 	encoderLayerNum = 10
# 	# 	decoderLayerNum = 10
# 	# 	channelGrowLayerNum = 5
# 	# 	ch = 2
# 	# 	layerNum = 20
# 	# 	innerLayerNum = 3

# 	# 	print('-------------------------G start----------------------------')
# 	# 	with tf.device(device[0]):
# 	# 		x = tf.layers.conv2d(x, 1, 3, strides = (1, 1), padding = 'valid', activation = tf.nn.relu, name = 'G_to510Conv_'+str(i), reuse = tf.AUTO_REUSE)
# 	# 		x = tf.nn.relu(x)
# 	# 		print('\tto510Conv:', x.shape)
# 	# 		assert x.shape == (x.shape[0], 510, 510, 1)

# 	# 	with tf.device(device[-1]):
# 	# 		y = x

# 	# 		y = tf.layers.average_pooling2d(inputs = y, pool_size = (3, 3), strides = (3, 3), padding = 'valid', data_format = 'channels_last', name = 'G_avgPooling_'+str(i))
# 	# 		print('\tafter_avg_pooling:', y.shape)
# 	# 		assert y.shape == (y.shape[0], 170, 170, 1)
# 	# 		y = tf.layers.flatten(y)
# 	# 		vecA = np.concatenate((y, np.ones(int(y.shape[0]))), axis = 0)
# 	# 		vecB = np.concatenate((np.ones(int(y.shape[0])), -y), axis = 0)

# 	# 		attMap = tf.matmul(vecA, vecB, transpose_b = True)
# 	# 		print('\tattMap:', attMap.shape)

# 	# 		attMap = tf.nn.softmax(attMap)


# 	# 		x00 = x[x.shape[0], 0::3, 0::3, 0]
# 	# 		x01 = x[x.shape[0], 0::3, 1::3, 0]
# 	# 		x02 = x[x.shape[0], 0::3, 2::3, 0]
# 	# 		x10 = x[x.shape[0], 1::3, 0::3, 0]
# 	# 		x11 = x[x.shape[0], 1::3, 1::3, 0]
# 	# 		x12 = x[x.shape[0], 1::3, 2::3, 0]
# 	# 		x20 = x[x.shape[0], 2::3, 0::3, 0]
# 	# 		x21 = x[x.shape[0], 2::3, 1::3, 0]
# 	# 		x22 = x[x.shape[0], 2::3, 2::3, 0]

# 	# 		z00 = tf.matmul(attMap, tf.layers.flatten(x00))
# 	# 		z01 = tf.matmul(attMap, tf.layers.flatten(x01))
# 	# 		z02 = tf.matmul(attMap, tf.layers.flatten(x02))
# 	# 		z10 = tf.matmul(attMap, tf.layers.flatten(x10))
# 	# 		z11 = tf.matmul(attMap, tf.layers.flatten(x11))
# 	# 		z12 = tf.matmul(attMap, tf.layers.flatten(x12))
# 	# 		z20 = tf.matmul(attMap, tf.layers.flatten(x20))
# 	# 		z21 = tf.matmul(attMap, tf.layers.flatten(x21))
# 	# 		z22 = tf.matmul(attMap, tf.layers.flatten(x22))

# 	# 		filter00 = np.reshape(np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter01 = np.reshape(np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter02 = np.reshape(np.array([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter10 = np.reshape(np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter11 = np.reshape(np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter12 = np.reshape(np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter20 = np.reshape(np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter21 = np.reshape(np.array([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)
# 	# 		filter22 = np.reshape(np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]), [1, 3, 3, 1]).transpose(1, 2, 3, 0)

# 	# 		h00 = tf.image.resize_images(images = tf.reshape(z00, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h01 = tf.image.resize_images(images = tf.reshape(z01, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h02 = tf.image.resize_images(images = tf.reshape(z02, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h10 = tf.image.resize_images(images = tf.reshape(z10, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h11 = tf.image.resize_images(images = tf.reshape(z11, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h12 = tf.image.resize_images(images = tf.reshape(z12, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h20 = tf.image.resize_images(images = tf.reshape(z20, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h21 = tf.image.resize_images(images = tf.reshape(z21, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)
# 	# 		h22 = tf.image.resize_images(images = tf.reshape(z22, shape=x00.shape), size = (x.shape[0], z00.shape[1]*3, z00.shape[2]*3, 1), method = 1)

# 	# 		h00 = tf.nn.conv2d(h00, filter = filter00, strides=(3, 3), padding='valid', name = 'NT_pickConv00')
# 	# 		h01 = tf.nn.conv2d(h01, filter = filter01, strides=(3, 3), padding='valid', name = 'NT_pickConv01')
# 	# 		h02 = tf.nn.conv2d(h02, filter = filter02, strides=(3, 3), padding='valid', name = 'NT_pickConv02')
# 	# 		h10 = tf.nn.conv2d(h10, filter = filter10, strides=(3, 3), padding='valid', name = 'NT_pickConv10')
# 	# 		h11 = tf.nn.conv2d(h11, filter = filter11, strides=(3, 3), padding='valid', name = 'NT_pickConv11')
# 	# 		h12 = tf.nn.conv2d(h12, filter = filter12, strides=(3, 3), padding='valid', name = 'NT_pickConv12')
# 	# 		h20 = tf.nn.conv2d(h20, filter = filter20, strides=(3, 3), padding='valid', name = 'NT_pickConv20')
# 	# 		h21 = tf.nn.conv2d(h21, filter = filter21, strides=(3, 3), padding='valid', name = 'NT_pickConv21')
# 	# 		h22 = tf.nn.conv2d(h22, filter = filter22, strides=(3, 3), padding='valid', name = 'NT_pickConv22')

# 	# 		h = h00 + h01 + h02 + h10 + h11 + h12 + h20 + h21 + h22

# 	# 		attOut = gamma * h + x

# 	# 		for i in range(encoderLayerNum):
# 	# 			x = tf.layers.conv2d(x, ch, 5, strides = (1, 1), padding = 'valid', activation = tf.nn.relu, name = 'G_encoConv_'+str(i), reuse = tf.AUTO_REUSE)
# 	# 			# x = batch_norm(x, is_training, scope='enco_batch_norm_' + str(i+1))
# 	# 			x = tf.nn.relu(x)
# 	# 			print('\tG_encoLayer:', x.shape)
# 	# 			ch = ch * 2 if i < channelGrowLayerNum else ch
# 	# 	E0x = x
# 	# 	print('------------------------------------------------------------')
		
# 	# 	with tf.device(device[0]):
# 	# 		# Self Attention
# 	# 		y = x
# 	# 		y = tf.layers.average_pooling2d(inputs = y, pool_size = (3, 3), strides = (3, 3), padding = 'valid', data_format = 'channels_last', name = 'G_innerEnco_'+str(i))
# 	# 		print('\tG_inner-encoLayer:', y.shape)
# 	# 		E1x = y

# 	# 		y, attMap = NNstructure_SAGAN.attentionBlock2(y, 64, sn=True, scope="attention", reuse=reuse)
# 	# 		print('\ty after attentionBlock:', y.shape)
# 	# 		E2x = y

# 	# 		# y = tf.layers.conv2d_transpose(inputs=y, filters=64, kernel_size=4, strides=(4, 4), padding='valid', use_bias=True)
# 	# 		y = tf.image.resize_images(images = y, size = (y.shape[1]*3+1, y.shape[2]*3+1), method = 1)
# 	# 		print('\tG_inner-decoLayer:', y.shape)
# 	# 		E3x = y

# 	# 		Wx = tf.get_variable('xWeight', shape=[x.shape[3]], dtype=tf.float32, trainable=True)
# 	# 		Wy = tf.get_variable('yWeight', shape=[y.shape[3]], dtype=tf.float32, trainable=True)
# 	# 		x = tf.nn.relu(x * Wx) + tf.nn.relu(y * Wy)
# 	# 		print('\tG_mergeLayer:', x.shape)

# 	# 	E4x = x
# 	# 	print('------------------------------------------------------------')
# 	# 	with tf.device(device[0]):
# 	# 		zz = list(range(decoderLayerNum - 1))
# 	# 		zz.reverse()
# 	# 		for i in zz:
# 	# 			x = SAGAN_ops.deconv(x, channels=ch, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='deconv_' + str(i))
# 	# 			# x = batch_norm(x, is_training, scope='deco_batch_norm_' + str(i))
# 	# 			x = tf.nn.relu(x)
# 	# 			print('\tG_decoLayer:', x.shape)
# 	# 			ch = ch // 2 if i < channelGrowLayerNum else ch

# 	# 		x = SAGAN_ops.deconv(x, channels=1, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='G_deconv_logit')
# 	# 		print('\tG_deconv-Final:', x.shape)
# 	# 		x = tf.tanh(x)
# 	# 		print('\tG_tanh:', x.shape)
# 	# 	print('---------------------------G end----------------------------')

# 	# 	return x, attMap, E0x, E1x, E2x, E3x, E4x, Wx, Wy








'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
from tensorflow.contrib import slim
batch_norm_params = {'decay': 0.995, 'epsilon': 0.001, 'updates_collections': None, 'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES]}

def _phase_shift(I, r, scope=None):
	with tf.variable_scope(scope):
		bsize, a, b, c = I.get_shape().as_list() # batchsize, 512, 512, cc/64
		bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
		X = tf.reshape(I, (bsize, a, b, r, r)) #(batchsize, 512, 512, r, r) r*r = cc/64
		X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, r, r
		X = tf.split(X, a, 1)  # a * (bsize, b, r, r)
		X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
		X = tf.split(X, b, 1)  # b * (bsize, a*r, r)
		X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
		return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False, scope=None):
	with tf.variable_scope(scope): # cc == 256
		if color:
			Xc = tf.split(X, 64, 3) # Xc.shape == (64, batchsize, 512, 512, cc/64)
			cnt = 0
			cX = []
			for x in Xc: # 64 times
				op_name = 'sp_{}'.format(cnt) # 0~63,     (batchsize, 512, 512, cc/64), r, 'sp_i'    cc/64==r*r
				cX.append(_phase_shift(x, r, op_name))  # 64 * (batchsize, 512*r, 512*r, 1)
				cnt += 1

			X = tf.concat(cX, 3) # (batchsize, 512, 512, 64)
			# X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
		else:
			X = _phase_shift(X, r)
	return X


def prelu(feature, scope='prelu'):
	with tf.variable_scope(scope):
		alphas = tf.get_variable(scope+'_alpha', feature.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pos = tf.nn.relu(feature)
		neg = alphas * (feature - tf.abs(feature)) * 0.5
	return pos + neg


def res_block(feature, kern_sz=3, channel_num=64, stride=1, weight_decay=0.05, scope=None):
	with tf.variable_scope(scope):
		net = slim.conv2d(feature, channel_num, [kern_sz, kern_sz], stride,
							weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
							weights_regularizer=slim.l2_regularizer(weight_decay),
							activation_fn=None)
		net = slim.batch_norm(net, param_initializers=batch_norm_params)
		net = prelu(net, scope)
		net = slim.conv2d(net, channel_num, [kern_sz, kern_sz], stride,
							weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
							weights_regularizer=slim.l2_regularizer(weight_decay),
							activation_fn=None)
		net = slim.batch_norm(net, param_initializers=batch_norm_params)
		net = net + feature
	return net

def generatorNN_814(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = ['/gpu:0']):

	feature, is_training, weight_decay, up_scale = NNinput, isTraining, 0.05, 4

	with tf.variable_scope(name):
		with slim.arg_scope([slim.batch_norm], is_training=is_training):
			with tf.device(device[0]):
				print('G: input:', feature.shape) # (8, 96, 96, 3)
				# k9n64s1 + PReLU
				net = slim.conv2d(feature, 64, [9, 9], activation_fn=None, scope='conv2d_1')
				net = prelu(net, 'prelu_1')
				print('G: conv+prelu:', net.shape) # (8, 96, 96, 64)

			with tf.device(device[-1]):
				# B residual blocks
				# k3n64s1 + BN + PReLU + k3n64s1 + BN
				resnet = net
				for blk_i in range(16):
					resnet = res_block(resnet, 3, 64, 1, weight_decay, 'resblock_{}'.format(blk_i))
					print('G: resblock:', resnet.shape) #(8, 96, 96, 64)

			with tf.device(device[0]):
				# k3n64s1 + BN
				resnet = slim.conv2d(resnet, 64, [3, 3], activation_fn=None, scope='conv2d_2')
				resnet = slim.batch_norm(resnet, param_initializers=batch_norm_params)
				print('G: resnet before shortcut:', resnet.shape) # (8, 96, 96, 64)
				net = net + resnet
				print('G: resnet after shortcut:', resnet.shape) # (8, 96, 96, 64)


				# subpixel
				spnet = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope='con2d_3_1')
				print('G: sub1-conv:', spnet.shape) # (8, 96, 96, 256)
				spnet = PS(spnet, 2, True, 'subpixel_3_1')
				print('G: sub1-PS:', spnet.shape) # (8, 192, 192, 64)
				spnet = prelu(spnet, 'prelu_3_1')
				print('G: sub1-prelu:', spnet.shape) # (8, 192, 192, 64)

				spnet = slim.conv2d(spnet, 256, [3, 3], activation_fn=None, scope='con2d_3_2')
				print('G: sub2-conv:', spnet.shape) # (8, 192, 192, 256)
				spnet = PS(spnet, 2, True, 'subpixel_3_2')
				print('G: sub2-PS:', spnet.shape) # (8, 384, 384, 64)
				spnet = prelu(spnet, 'prelu_3_2')
				print('G: sub2-prelu:', spnet.shape) # (8, 384, 384, 64)

				# k9n3s1
				net = slim.conv2d(spnet, 1, [9, 9], activation_fn=tf.nn.tanh, scope='conv2d_4')
				print('G: output:', net.shape) #(8, 384, 384, 3)

	# return net
	# return tf.reshape(net, (net.shape[0], 512, 512, 1))
	return tf.image.resize_images(images = net, size = (512, 512), method = 1), net



'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''

# def generatorNN_1018(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'G_testGenerator', device = ['/gpu:0']):
# 	print('Info: NNstructure.generatorNN_721NSA() start')
# 	with tf.variable_scope(name, reuse = reuse):
# 		print('\tNNinput =', NNinput.shape)
# 		x = NNinput
# 		encoderLayerNum = 10
# 		decoderLayerNum = 10
# 		channelGrowLayerNum = 5
# 		ch = 2
# 		layerNum = 20
# 		innerLayerNum = 3

# 		midList = []

# 		print('-------------------------G start----------------------------')
# 		with tf.device(device[0]):
# 			for i in range(encoderLayerNum):
# 				x = tf.layers.conv2d(x, ch, 5, strides = (1, 1), padding = 'valid', activation = tf.nn.relu, name = 'G_encoConv_'+str(i), reuse = tf.AUTO_REUSE)
# 				# x = batch_norm(x, is_training, scope='enco_batch_norm_' + str(i+1))
# 				x = tf.nn.relu(x)
# 				midList.append(x)
# 				print('\tG_encoLayer:', x.shape)
# 				ch = ch * 2 if i < channelGrowLayerNum else ch
# 		E0x = x
# 		print('------------------------------------------------------------')
# 		E4x = x
# 		print('------------------------------------------------------------')
# 		with tf.device(device[0]):
# 			zz = list(range(decoderLayerNum - 1))
# 			zz.reverse()
# 			for i in zz:
# 				preX = midList.pop()
# 				x = tf.concat([x, preX], axis = 3)
# 				x = SAGAN_ops.deconv(x, channels=ch, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='deconv_' + str(i))
# 				# x = batch_norm(x, is_training, scope='deco_batch_norm_' + str(i))
# 				x = tf.nn.relu(x)
# 				print('\tG_decoLayer:', x.shape)
# 				ch = ch // 2 if i < channelGrowLayerNum else ch

# 			x = SAGAN_ops.deconv(x, channels=1, kernel=5, stride=1, padding = 'valid', use_bias=False, sn=True, scope='G_deconv_logit')
# 			print('\tG_deconv-Final:', x.shape)
# 			x = tf.tanh(x)
# 			print('\tG_tanh:', x.shape)
# 		print('---------------------------G end----------------------------')

# 		return x, None, E0x, None, None, None, E4x, None, None


'''=============================================================================================================='''
'''=============================================================================================================='''

# def discriminatorNN_1018(NNinput, isTraining, reuse = tf.AUTO_REUSE, name = 'D_testDiscriminator', device = '/gpu:1'):
# 	print('Info: NNstructure.discriminatorNN_21NSA() start.')
# 	with tf.device(device):
# 		with tf.variable_scope(name, reuse = reuse):
# 			print('---------------------------D start-----------------------------')

# 			print('\tD_NNinput:', NNinput.shape)

# 			data = NNinput
# 			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv1', reuse = tf.AUTO_REUSE)
# 			print('\tD_convLayer:', data.shape)
# 			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv2', reuse = tf.AUTO_REUSE)
# 			print('\tD_convLayer:', data.shape)
# 			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv3', reuse = tf.AUTO_REUSE)
# 			print('\tD_convLayer:', data.shape)
# 			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv4', reuse = tf.AUTO_REUSE)
# 			print('\tD_convLayer:', data.shape)
# 			data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv5', reuse = tf.AUTO_REUSE)
# 			print('\tD_convLayer:', data.shape)
# 			data = tf.layers.conv2d(data, 1, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv6', reuse = tf.AUTO_REUSE)
# 			print('\tD_convLayer:', data.shape)
# 			# data = tf.layers.conv2d(data, 5, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.relu, name = 'D_conv7', reuse = tf.AUTO_REUSE)
# 			# print('\tD_convLayer:', data.shape)
# 			# data = tf.layers.conv2d(data, 1, 3, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv8', reuse = reuse)
# 			# print('\tD_convLayer:', data.shape)
# 			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv9', reuse = reuse)
# 			# data = tf.layers.conv2d(data, 1, 7, strides = (2, 2), padding = 'valid', activation = tf.nn.sigmoid, name = 'D_conv10', reuse = reuse)
# 			data = slim.flatten(data)
# 			print('\tD_flattenLayer:', data.shape)
# 			data = layerBlocks.lrelu(data, 0.2)
# 			# data = layerBlocks.denselayer(data, 1)
# 			data = tf.nn.sigmoid(data)
# 			# data = tf.squeeze(data)
# 			data = tf.reduce_sum(data)
# 			print('\tD_output:', data.shape)
# 			print('------------------------------D finish-----------------------------')

# 	return data




'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
# def generatorNN_1107(NNinput, isTraining = True, reuse = tf.AUTO_REUSE, name = 'G_NN1107', device = ['/gpu:1']):
# 	import NNstructure_GAN_CIRCLE
# 	with tf.device(device[0]):
# 		return NNstructure_GAN_CIRCLE.generator(x = NNinput, nf = 32, c = 1, scope = name)


# def discriminatorNN_1107(NNinput, isTraining = True, reuse = tf.AUTO_REUSE, name = 'D_NN1107', device = '/gpu:1'):
# 	import NNstructure_GAN_CIRCLE
# 	with tf.device(device):
# 		return NNstructure_GAN_CIRCLE.discriminator(x = NNinput, nf = 64, scope = name)