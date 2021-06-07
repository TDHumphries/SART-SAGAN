
import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.variance_scaling_initializer()
weight_regularizer = None
# weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=1, padding='same', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels, padding = padding, kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, strides=stride, use_bias=use_bias)

    return x

def deconv(x, channels, kernel=3, stride=2, use_bias=True, scope='deconv_0') :
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding='SAME')

    return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

    return x

def flatten(x) :
    return tf.layers.flatten(x)

def resize(image, scale):
	num_imgs,height,width = image.shape[0],image.shape[1],image.shape[2]
	new_width = int(width * scale)
	new_height = int(height * scale)
	curr_img = image
	curr_img = tf.image.resize_bicubic(curr_img, [new_height, new_width])
	return curr_img

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

    return x + x_init

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake):
    real_loss = 0
    fake_loss = 0

    if type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if type == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if type == 'wgan':
        real_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
        fake_loss = 0

    loss = real_loss + fake_loss

    return loss


def generator_loss(type, fake):
    fake_loss = 0

    if type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if type == 'gan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if type == 'wgan':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss


    return loss


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

'''==========================================================================================================================='''

relu = tf.nn.relu

def lrelu(x, leak=0.2, scope='lrelu'):
    with tf.variable_scope(scope):
        return tf.maximum(x, leak * x)

def conv2d(x, n_outputs, kernel_size, stride=1, padding='SAME', 
           activation_fn=None, scope='conv2d'):
    with tf.variable_scope(scope):
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        return tf.contrib.layers.conv2d(x, n_outputs, kernel_size, stride, 
                                        padding, activation_fn=activation_fn, 
                                        weights_initializer=w_init, 
                                        biases_initializer=b_init)

def deconv2d(x, n_outputs, kernel_size, stride=1, padding='SAME', 
           activation_fn=None, scope='deconv2d'):
    with tf.variable_scope(scope):
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        return tf.contrib.layers.conv2d_transpose(x, n_outputs, 
                                                  [kernel_size, kernel_size], 
                                                  [stride, stride], padding, 
                                                  activation_fn=activation_fn, 
                                                  weights_initializer=w_init, 
                                                  biases_initializer=b_init)

def bn(x, eps=1e-5, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, 
                                        epsilon=eps, scale=True, scope=scope)

def instance_norm(x, eps=1e-5, scope='instance_norm'):
    with tf.variable_scope(scope):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], 
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + eps)) + offset
        return out



'''========================================================================================================================================='''


def resnet_block(x, nf, scope='res'):
    with tf.variable_scope(scope):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        y = conv2d(y, nf, 3, 1, padding='VALID', scope='_conv1')
        y = instance_norm(y, scope='_norm1')
        y = relu(y, name='_relu1')
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        y = conv2d(y,nf, 3, 1, padding='VALID', scope='_conv2')
        y = instance_norm(y, scope='_norm2')
        return x + y

def generator(x, nf=32, c=1, scope='gen'):
    channel = 64
    layers = 12
    scale = 2
    min_filters = 16
    filters_decay_gamma = 1.5
    dropoutRate = 0.8
    padding = 'same'
    reuse=tf.AUTO_REUSE
    if scope == 'g_B':
        pixel_shuffler = True
        G_B = False
        n_stride = 2
    else:
        pixel_shuffler = True
        G_B = False
        n_stride = 2
        # channel = channel//2 
    he_init = tf.variance_scaling_initializer()
    
    with tf.variable_scope(scope, reuse=reuse):
        #bicubic image
        if G_B:
            bicubic_img = resize(x, scale)
        channel = channel * n_stride

        #feature map
        feature = conv(x, channel, kernel=3, stride=n_stride, scope='conv_0')   
        feature = lrelu(feature, 0.1)
        #print("feature shape", feature.shape)

        if dropoutRate < 1:
            feature = tf.layers.dropout(feature, dropoutRate)
        inputs = tf.identity(feature)
        #print("inputs shape", inputs.shape)

        #Dense layers
        for i in range(1, layers):
            if min_filters != 0 and i > 0:
                x1 = i / float(layers -1)
                y1 = pow(x1, 1.0 / filters_decay_gamma)
                output_feature_num = int((channel - min_filters) * (1 - y1) + min_filters)

            inputs = conv(inputs, output_feature_num, kernel=3,scope='conv'+str(i))
            # inputs = instance_norm(inputs, scope='ins_'+str(i))
            inputs = lrelu(inputs, 0.1)
            if dropoutRate < 1:
                inputs = tf.layers.dropout(inputs, dropoutRate)
            feature = tf.concat([feature, inputs], -1)
        
        #Reconstruction layers
        recons_a = tf.layers.conv2d(feature, 64, 1, padding='same', kernel_initializer=he_init,  name='A1', use_bias = True)
        recons_a = tf.nn.leaky_relu(recons_a, 0.1)

        if dropoutRate < 1:
            recons_a = tf.layers.dropout(recons_a, dropoutRate)
    
        recons_b1 = tf.layers.conv2d(feature, 32, 1, padding=padding, kernel_initializer=he_init, name='B1', use_bias = True)
        recons_b1 = tf.nn.leaky_relu(recons_b1, 0.1)
        if dropoutRate < 1:
            recons_b1 = tf.layers.dropout(recons_b1, dropoutRate)
    
        recons_b2 = tf.layers.conv2d(recons_b1, 8, 3, padding=padding, kernel_initializer=he_init, name='B2', use_bias = True)
        if dropoutRate < 1:
            recons_b = tf.layers.dropout(recons_b2, dropoutRate)
    
        recons = tf.concat([recons_a, recons_b], -1)
        num_feature_map = recons.get_shape()[-1]

        #building upsampling layer
        if pixel_shuffler:
            if scale == 4:
                recons_ps1 = pixelShuffler(recons, scale)
                recons = pixelShuffler(recons_ps1, scale)
            else:
                recons = tf.layers.conv2d(recons, num_feature_map, 3, padding=padding, kernel_initializer=he_init, name='Up-PS.conv0', use_bias = True)
                recons = tf.layers.conv2d_transpose(recons, num_feature_map//2, 4, strides = 2, padding=padding, kernel_initializer=he_init, name='Up-PS.T')
                recons = lrelu(recons, 0.1)
                
        out = tf.layers.conv2d(recons, 1, 3, padding=padding, kernel_initializer=he_init, name='R.conv0', use_bias = False)
        
        out = x + out

    return out

def discriminator(x, nf=64, scope='dis'):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        #Convolutional layers
        d_c1 = conv2d(x, nf, 4, stride=2, scope='conv1')
        d_r1 = lrelu(d_c1, scope='lrelu1')
        
        d_c2 = conv2d(d_r1, nf * 2, 4, stride=2, scope='conv2')
        d_n2 = instance_norm(d_c2, scope='norm2')
        d_r2 = lrelu(d_n2, scope='lrelu2')
        
        d_c3 = conv2d(d_r2, nf * 4, 4, stride=2, scope='conv3')
        d_n3 = instance_norm(d_c3, scope='norm3')
        d_r3 = lrelu(d_n3, scope='lrelu3')
        
        d_c4 = conv2d(d_r3, nf * 8, 4, stride=1, scope='conv4')
        d_n4 = instance_norm(d_c4, scope='norm4')
        d_r4 = lrelu(d_n4, scope='lrelu4')
        
        d_c5 = conv2d(d_r4, 1, 4, stride=1, scope='conv5')
        
        # return d_c5
        data = tf.layers.dense(d_c5, 1)
        data = tf.reduce_sum(data)
        print('CIRCLE_D: data.shape:', data.shape)
        return data


'''============================================================================================================='''
'''============================================================================================================='''
'''============================================================================================================='''
'''============================================================================================================='''
'''============================================================================================================='''
'''============================================================================================================='''

def cyclic_loss_new(real, cycle):
    cost = tf.reduce_sum(tf.abs(real - cycle))/(2.0 * 64)
    return cost

def cyclic_loss(real, cycle):
    return tf.reduce_mean(tf.abs(real - cycle))

def lsgan_gen_loss(fake):
    return tf.reduce_mean(tf.squared_difference(fake, 1))

def lsgan_dis_loss(real, fake):
    return (tf.reduce_mean(tf.squared_difference(real, 1)) + 
            tf.reduce_mean(tf.squared_difference(fake, 0))) * 0.5

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake):
    real_loss = 0
    fake_loss = 0

    if type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if type == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if type == 'wgan':
        real_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
        fake_loss = 0

    loss = real_loss + fake_loss

    return loss


def generator_loss(type, fake):
    fake_loss = 0

    if type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if type == 'gan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if type == 'wgan':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss


    return loss


'''======================================================================================================='''

def zz_g_loss(X, Y, Gx, Fy):
    # real_a = X
    # real_b = Y
    # cycle_a = Fy
    # cycle_b = Gx
    # #Cycle consistency loss
    # cyclic_loss_a = 1.0 * cyclic_loss_new(real_a, cycle_a)
    # cyclic_loss_b = 1.0 * cyclic_loss_new(real_b, cycle_b)
    
    # #LSGAN loss
    # lsgan_loss_a = generator_loss(type = "wgan", fake = p_fake_a)
    # lsgan_loss_b = generator_loss(type = "wgan", fake = self.p_fake_b)


    # #Identity loss
    # identity_loss_a = 1.0 * cyclic_loss_new(self.real_a, self.p_fake_aa)/2
    # identity_loss_b = 1.0 * cyclic_loss_new(self.real_b, self.p_fake_bb)/2

    # #Identity loss
    # joint_loss_a = 1.0 / 10 * tf.reduce_sum(tf.image.total_variation(p_fake_aa))/20 + (1-self.lambda_a / 10) * tf.reduce_sum(tf.image.total_variation(real_a - p_fake_aa))/20
    # joint_loss_b = 1.0 / 10 * tf.reduce_sum(tf.image.total_variation(p_fake_bb))/20 + (1-self.lambda_b / 10) * tf.reduce_sum(tf.image.total_variation(real_b - p_fake_bb))/20

    # #Generator loss
    # g_a_loss = cyclic_loss_a + cyclic_loss_b + lsgan_loss_b + identity_loss_a + joint_loss_a
    # g_b_loss = cyclic_loss_b + cyclic_loss_a + lsgan_loss_a + identity_loss_b + joint_loss_b

    # g_loss = g_a_loss + g_b_loss

    return None