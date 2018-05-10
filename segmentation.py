import datetime
import math
import os
import random
import sys
from os import listdir

import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
from PIL import Image
from skimage import img_as_float

# from tensorflow.python.framework import ops

NUM_CLASSES = 2


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_params(list_params):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in xrange(1, len(sys.argv)):
        print list_params[i - 1] + '= ' + sys.argv[i]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


def normalize_images(data, mean_full, std_full):
    data[:, :, :, 0] = np.subtract(data[:, :, :, 0], mean_full[0])
    data[:, :, :, 1] = np.subtract(data[:, :, :, 1], mean_full[1])
    data[:, :, :, 2] = np.subtract(data[:, :, :, 2], mean_full[2])

    data[:, :, :, 0] = np.divide(data[:, :, :, 0], std_full[0])
    data[:, :, :, 1] = np.divide(data[:, :, :, 1], std_full[1])
    data[:, :, :, 2] = np.divide(data[:, :, :, 2], std_full[2])


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def load_images(path, specific_event=None):
    images = []
    masks = []

    for d in listdir(path):
        if "tar.gz" not in d and ".py" not in d and "npy" not in d and "zip" not in d and "output" not in d and \
                        "txt" not in d and "new" not in d and "old" not in d and "test" != d and \
                        "sh" not in d and "aux" not in d and "pkl" not in d:
            if specific_event is None or (specific_event is not None and specific_event == int(d.split("_")[1])):
                print BatchColors.WARNING + "Reading event " + d.split("_")[1] + BatchColors.ENDC
                for f in listdir(path + d):
                    if "tif" not in f and ".png.aux.xml" not in f:
                        try:
                            if 'mask' in d:
                                img = scipy.misc.imread(path + d + '/' + f)
                                masks.append((int(f[9:15]), img))
                            # print f, f[9:15], type(img), img.shape, np.bincount(img.flatten())
                            else:
                                img = img_as_float(scipy.io.loadmat(path + d + '/' + f)['img'])
                                images.append((int(f[:-4]), img))
                                # print f, f[:-4], type(img), img.shape, np.max(img), np.min(img)
                        except IOError:
                            print BatchColors.FAIL + "Could not open/read file: " + path + d + '/' + f + BatchColors.ENDC

    masks.sort(key=lambda tup: tup[0])
    images.sort(key=lambda tup: tup[0])

    return images, masks


def dynamically_calculate_mean_and_std(data, crop_size, class_distribution):
    total_length = len(class_distribution[0]) + len(class_distribution[1])
    total_class_distribution = class_distribution[0] + class_distribution[1]

    patches = []
    for i in xrange(total_length):
        cur_map = total_class_distribution[i][0]
        cur_x = total_class_distribution[i][1][0]
        cur_y = total_class_distribution[i][1][1]
        # curTransform = total_class_distribution[i][2]

        patch = data[cur_map][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        if len(patch) != crop_size or len(patch[0]) != crop_size:
            print "Error: Current patch size ", len(patch), len(patch[0])
            return
        patches.append(patch)
    return compute_image_mean(np.asarray(patches))


def create_balanced_validation_set(class_distribution, percentage_val=0.2):
    validation_distribution = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    for i in xrange(len(class_distribution)):
        for j in xrange(len(class_distribution[i]) - 1, -1, -1):  # range(100,-1,-1)
            if random.randint(1, 10) > 10 - percentage_val * 10:
                validation_distribution[i].append(class_distribution[i][j])
                del class_distribution[i][j]

    return validation_distribution


def create_distributions_over_classes(labels, crop_size, stride_crop):
    classes = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    for k in xrange(len(labels)):
        w, h = labels[k][1].shape

        for i in xrange(0, w, stride_crop):
            for j in xrange(0, h, stride_crop):
                cur_x = i
                cur_y = j
                patch_class = labels[k][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[k][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                # print '1', len(patch_class), len(patch_class[0])
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = labels[k][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                # print '2', len(patch_class), len(patch_class[0])
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[k][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                # print j, cur_y, crop_size, len(patch_class[0]), (crop_size-len(patch_class[0]))
                # print '3', len(patch_class), len(patch_class[0])

                if patch_class.shape == (crop_size, crop_size):
                    count = np.bincount(patch_class.astype(int).flatten())
                    classes[int(np.argmax(count))].append((k, (cur_x, cur_y)))
                else:
                    print BatchColors.FAIL + 'Error: size of patch_class is ' + patch_class.shape + BatchColors.ENDC

    return classes


def dynamically_create_patches(data, mask_data, crop_size, class_distribution, shuffle):
    patches = []
    classes = []

    for i in shuffle:
        if i >= 2 * len(class_distribution):
            cur_pos = i - 2 * len(class_distribution)
        elif i >= len(class_distribution):
            cur_pos = i - len(class_distribution)
        else:
            cur_pos = i

        cur_map = class_distribution[cur_pos][0]
        cur_x = class_distribution[cur_pos][1][0]
        cur_y = class_distribution[cur_pos][1][1]
        # curTransform = class_distribution[cur_pos][2]
        # print cur_map, cur_x, cur_y

        patch = data[cur_map][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        current_class = mask_data[cur_map][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

        if len(patch) != crop_size or len(patch[0]) != crop_size:
            print "Error: Current patch size ", len(patch), len(patch[0])
            print cur_map, cur_x, cur_y
            return

        if i < len(class_distribution):
            patches.append(patch)
            classes.append(current_class)
        elif i < 2 * len(class_distribution):
            patches.append(np.fliplr(patch))
            classes.append(np.fliplr(current_class))
        elif i >= 2 * len(class_distribution):
            patches.append(np.flipud(patch))
            classes.append(np.flipud(current_class))

    return np.asarray(patches), np.asarray(classes, dtype=np.int32)


def convert_rgb_to_class(crop):
    class_map = np.empty([len(crop), len(crop[0])], dtype=np.uint8)
    count = 5 * [0]
    for i in xrange(len(crop)):
        for j in xrange(len(crop[0])):
            val = crop[i, j, :]
            if val[0] == 0 and val[1] == 0 and val[2] == 0:
                current_class = 4
            elif val[0] == 0 and val[1] == 0 and val[2] == 255:
                current_class = 1
            elif val[0] == 255 and val[1] == 0 and val[2] == 0:
                current_class = 2
            elif val[0] == 0 and val[1] == 255 and val[2] == 0:
                current_class = 3
            elif val[0] == 0 and val[1] == 255 and val[2] == 255:
                current_class = 0
            else:
                print("ERROR: Class not found! ", val)
                current_class = -1
            class_map[i][j] = current_class
            count[current_class] += 1
    return class_map, np.argmax(count)


def save_map(path, prob_im_argmax, data):
    for i in xrange(len(data)):
        name = format(data[i][0], '06')

        img = Image.fromarray(np.uint8(prob_im_argmax[i] * 255))
        img.save(path + name + '.png')

        scipy.misc.toimage(prob_im_argmax[i], cmin=0.0, cmax=255).save(path + 'seg_mask_' + name + '.png')


def calc_accuracy_by_crop(true_crop, pred_crop, track_conf_matrix):
    b, h, w = pred_crop.shape
    _trueCrop = np.reshape(true_crop, (b, h, w))

    acc = 0
    local_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    # count = 0
    for i in xrange(b):
        for j in xrange(h):
            for k in xrange(w):
                # count += 1
                if _trueCrop[i][j][k] == pred_crop[i][j][k]:
                    acc = acc + 1
                track_conf_matrix[_trueCrop[i][j][k]][pred_crop[i][j][k]] += 1
                local_conf_matrix[_trueCrop[i][j][k]][pred_crop[i][j][k]] += 1

    # print 'count', count
    return acc, local_conf_matrix


'''
TensorFlow
'''


def _add_weight_decay(var, weight_decay, collection_name='losses'):
    if not tf.get_variable_scope().reuse:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return var


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def _variable_on_cpu(name, shape, ini):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=ini, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, ini, weight_decay):
    var = _variable_on_cpu(name, shape, ini)
    # tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    # tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if weight_decay is not None:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _max_pool(input_data, kernel, strides, name, pad='SAME'):
    return tf.nn.max_pool(input_data, ksize=kernel, strides=strides, padding=pad, name=name)


def _max_pool_with_argmax(input_data, kernel, strides, name):
    pool, argmax = tf.nn.max_pool_with_argmax(input_data, ksize=kernel, strides=strides, padding='SAME', name=name)
    return pool, argmax


def _batch_norm(input_data, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=True, center=False,
                                                        updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=False, center=False,
                                                        updates_collections=None, scope=scope, reuse=True)
                   )


def _conv_layer(input_data, kernel_shape, name, weight_decay, is_training, rate=1, strides=None, pad='SAME',
                activation='relu', has_batch_norm=True, has_activation=True, is_standard_conv=False):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=kernel_shape,
                                              ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', kernel_shape[-1], tf.constant_initializer(0.1))

        if is_standard_conv is False:
            conv_op = tf.nn.atrous_conv2d(input_data, weights, rate=rate, padding=pad)
        else:
            conv_op = tf.nn.conv2d(input_data, weights, strides=strides, padding=pad)
        conv_act = tf.nn.bias_add(conv_op, biases)

        if has_batch_norm is True:
            conv_act = _batch_norm(conv_act, is_training, scope=scope)
        if has_activation is True:
            if activation == 'relu':
                conv_act = tf.nn.relu(conv_act, name=name)
            else:
                conv_act = leaky_relu(conv_act)

        return conv_act


def _deconv_filter(final_shape):
    width = final_shape[0]
    height = final_shape[1]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([final_shape[0], final_shape[1]])
    for x in xrange(width):
        for y in xrange(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(final_shape)
    for i in range(final_shape[2]):
        for j in range(final_shape[3]):
            weights[:, :, i, j] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


def _deconv_layer(input_data, f_shape, output_shape, is_training, stride=2, has_batch_norm=True, has_bias=True,
                  has_activation=True, weight_decay=None, name=None):
    # output_shape = [b, w, h, c]
    strides = [1, stride, stride, 1]

    # output_shape = tf.stack([layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3]])

    with tf.variable_scope(name) as scope:
        # in_features = input_data.get_shape()[3].value

        weights = _deconv_filter(f_shape)
        biases = _variable_on_cpu('biases', f_shape[2], tf.constant_initializer(0.1))
        if weight_decay is not None:
            _add_weight_decay(weights, weight_decay)

        deconv_op = tf.nn.conv2d_transpose(input_data, weights, output_shape, strides=strides, padding='SAME')
        if has_bias is True:
            deconv_op = tf.nn.bias_add(deconv_op, biases)
        if has_batch_norm is True:
            deconv_op = _batch_norm(deconv_op, is_training, scope=scope.name)
        if has_activation is True:
            deconv_op = leaky_relu(deconv_op)

    return deconv_op


'''
NETWORKS
'''

'''
segnet_25
'''


def deconvnet_1(x, dropout, is_training, weight_decay, crop_size, channels):
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # 50x50xchannels

    # norm1
    norm1 = tf.nn.local_response_normalization(x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                                               name='norm1')  # 50x50xchannels

    # conv1
    # conv0 = _conv_layer(norm1, [7, 7, channels, 64], "conv0", weight_decay, is_training) ## 50x50x64
    # pool1
    # pool0, pool0_indices = _max_pool_with_argmax(conv0, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool0')
    # 25x25x64

    # conv1
    conv1 = _conv_layer(norm1, [5, 5, channels, 64], "conv1", weight_decay, is_training)  # 25x25x128
    # pool1
    pool1, pool1_indices = _max_pool_with_argmax(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                 name='pool1')  # 13x13x64

    # conv2
    conv2 = _conv_layer(pool1, [3, 3, 64, 128], "conv2", weight_decay, is_training)  # 13x13x128
    # pool2
    pool2, pool2_indices = _max_pool_with_argmax(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                 name='pool2')  # 7x7x128

    # conv3
    conv3 = _conv_layer(pool2, [3, 3, 128, 256], "conv3", weight_decay, is_training)  # 7x7x256
    # pool3
    pool3, pool3_indices = _max_pool_with_argmax(conv3, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                 name='pool3')  # 4x4x256

    """ --------------------------------------End of encoder------------------------------------------ """

    """ start upsample """

    # upsample 3
    upsample3 = _deconv_layer(pool3, f_shape=[2, 2, 128, 256], output_shape=tf.shape(pool2), is_training=is_training,
                              stride=2, has_batch_norm=False, has_bias=False, has_activation=False,
                              weight_decay=weight_decay,
                              name="up3")  # 7x7x128
    # decode 3
    conv_decode3 = _conv_layer(upsample3, [3, 3, 128, 128], "conv_decode3", weight_decay, is_training,
                               has_activation=False)  # 7x7x128

    # upsample2
    upsample2 = _deconv_layer(conv_decode3, f_shape=[2, 2, 64, 128], output_shape=tf.shape(pool1),
                              is_training=is_training, stride=2, has_batch_norm=False, has_bias=False,
                              has_activation=False,
                              weight_decay=weight_decay, name="up2")  # 13x13x128
    # decode 2
    conv_decode2 = _conv_layer(upsample2, [3, 3, 64, 64], "conv_decode2", weight_decay, is_training,
                               has_activation=False)  # 13x13x128

    # upsample1
    try:
        out = tf.pack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])
    except:
        out = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])  # tf.shape(pool0)
    upsample1 = _deconv_layer(conv_decode2, f_shape=[2, 2, 64, 64], output_shape=out, is_training=is_training, stride=2,
                              has_batch_norm=False, has_bias=False, has_activation=False, weight_decay=weight_decay,
                              name="up1")  # 25x25x64
    # decode1
    conv_decode1 = _conv_layer(upsample1, [5, 5, 64, 64], "conv_decode1", weight_decay, is_training,
                               has_activation=False)  # 25x25x64

    # upsample0
    # try:
    # out = tf.pack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])
    # except:
    # out = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])
    # upsample0 = _deconv_layer(conv_decode1, f_shape=[2, 2, 64, 64], output_shape=out, is_training=is_training,
    # stride=2, has_batch_norm=False, has_bias=False, has_activation=False, weight_decay=weight_decay, name="up0")
    #  50x50x64
    # decode0
    # conv_decode0 = _conv_layer(upsample0, [7, 7, 64, 64], "conv_decode0", weight_decay, is_training,
    # has_activation=False) ## 50x50x64
    """ end of Decode """

    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


'''
segnet_icpr
'''


def deconvnet_2(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 50x50xchannels
    # print x.get_shape()

    # conv0 = _conv_layer(x, [4,4,channels,64], "conv0", weight_decay, is_training) # 50x50x64
    # pool0, pool0_indices = _max_pool_with_argmax(conv0, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool0')
    # 25x25x64

    conv1 = _conv_layer(x, [4, 4, channels, 64], "conv1", weight_decay, is_training)  # 25x25x64 pool0
    pool1, pool1_indices = _max_pool_with_argmax(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                 name='pool1')  # 13x13x64

    conv2 = _conv_layer(pool1, [4, 4, 64, 128], 'conv2', weight_decay, is_training)  # 13x13x128
    pool2, pool2_indices = _max_pool_with_argmax(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                 name='pool2')  # 7x7x128

    conv3 = _conv_layer(pool2, [3, 3, 128, 256], 'conv3', weight_decay, is_training)  # 7x7x256
    pool3, pool3_indices = _max_pool_with_argmax(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                                 name='pool3')  # 4x4x256

    # upsample 3
    upsample3 = _deconv_layer(pool3, f_shape=[3, 3, 256, 256], output_shape=tf.shape(conv3), is_training=is_training,
                              stride=1, has_batch_norm=False, has_bias=False, has_activation=False,
                              weight_decay=weight_decay,
                              name="up3")  # 7x7x256
    # decode 3
    conv_decode3 = _conv_layer(upsample3, [3, 3, 256, 128], "conv_decode3", weight_decay, is_training,
                               has_activation=False)  # 7x7x128

    # upsample2
    upsample2 = _deconv_layer(conv_decode3, f_shape=[3, 3, 128, 128], output_shape=tf.shape(conv2),
                              is_training=is_training, stride=2, has_batch_norm=False, has_bias=False,
                              has_activation=False,
                              weight_decay=weight_decay, name="up2")  # 13x13x128
    # decode 2
    conv_decode2 = _conv_layer(upsample2, [4, 4, 128, 128], "conv_decode2", weight_decay, is_training,
                               has_activation=False)  # 13x13x128

    # upsample1
    try:
        out = tf.pack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])
    except:
        out = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])  # tf.shape(conv1)
    upsample1 = _deconv_layer(conv_decode2, f_shape=[3, 3, 64, 128], output_shape=out, is_training=is_training,
                              stride=2, has_batch_norm=False, has_bias=False, has_activation=False,
                              weight_decay=weight_decay,
                              name="up1")  # 25x25x128
    # decode4
    conv_decode1 = _conv_layer(upsample1, [3, 3, 64, 64], "conv_decode1", weight_decay, is_training,
                               has_activation=False)  # 25x25x64

    # upsample0
    # try:
    # out = tf.pack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])
    # except:
    # out = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 64])
    # upsample0 = _deconv_layer(conv_decode1, f_shape=[3,3,64,64], output_shape=out, is_training=is_training, 
    # stride=2, has_batch_norm=False, has_bias=False, has_activation=False, weight_decay=weight_decay, name="up0") 
    # 50x50x64
    # decode4
    # conv_decode0 = _conv_layer(upsample0, [3,3,64,64], "conv_decode0", weight_decay, is_training,
    # has_activation=False) # 50x50x64
    """ end of Decode """

    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


'''
dilated_icpr
'''


def dilated_convnet_1(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25
    # print x.get_shape()

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1)
    # pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=1)
    # pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=2)
    # pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=2)
    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=4)
    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=4)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


'''
dilated_grsl
'''


def dilated_convnet_2(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25
    # print x.get_shape()

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool1')

    conv2 = _conv_layer(pool1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2, activation='lrelu')
    pool2 = _max_pool(conv2, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool2')

    conv3 = _conv_layer(pool2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=3, activation='lrelu')
    pool3 = _max_pool(conv3, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(pool3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4, activation='lrelu')
    pool4 = _max_pool(conv4, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool4')

    conv5 = _conv_layer(pool4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5, activation='lrelu')
    pool5 = _max_pool(conv5, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool5')

    conv6 = _conv_layer(pool5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6, activation='lrelu')
    pool6 = _max_pool(conv6, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool6')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def loss_def(_logits, _labels):
    logits = tf.reshape(_logits, [-1, NUM_CLASSES])
    labels = tf.cast(tf.reshape(_labels, [-1]), tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def validate(sess, data, labels, validation_class_distribution, n_input_data, n_input_data_mask, batch_size, x, y,
             keep_prob, is_training, pred_up, upscore, step, crop_size, mean, std, output_path):
    if len(validation_class_distribution) == 2:
        validation_class_distribution = validation_class_distribution[0] + validation_class_distribution[1]
    total_length = len(validation_class_distribution)

    all_predcs = np.empty([total_length, crop_size, crop_size], dtype=np.uint8)
    all_classes = np.empty([total_length, crop_size, crop_size], dtype=np.uint32)
    cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

    prob_im = np.zeros([total_length, crop_size, crop_size, NUM_CLASSES], dtype=np.float32)

    list_index = np.arange(total_length)

    for i in xrange(0,
                    ((total_length / batch_size) if (total_length % batch_size) == 0 else (
                                total_length / batch_size) + 1)):
        b_x, b_y = dynamically_create_patches(data, labels, crop_size, validation_class_distribution,
                                              list_index[i * batch_size:min(i * batch_size + batch_size, total_length)])
        normalize_images(b_x, mean, std)

        batch_x = np.reshape(b_x, (-1, n_input_data))
        batch_y = np.reshape(b_y, (-1, n_input_data_mask))

        _pred_up, _upscore = sess.run([pred_up, upscore],
                                      feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})

        all_predcs[i * batch_size:min(i * batch_size + batch_size, total_length), :, :] = _pred_up
        all_classes[i * batch_size:min(i * batch_size + batch_size, total_length), :, :] = b_y

        prob_im[i * batch_size:min(i * batch_size + batch_size, total_length), :, :, :] = _upscore

    epoch_mean, _ = calc_accuracy_by_crop(all_classes, all_predcs, cm_test)
    np.save(output_path + 'prob_im.npy', prob_im)

    # Norm Accuracy
    _sum = 0.0
    for i in xrange(len(cm_test)):
        _sum += (cm_test[i][i] / float(np.sum(cm_test[i])) if np.sum(cm_test[i]) != 0 else 0)

    # IoU Flooding
    _sum_iou = (cm_test[1][1] / float(np.sum(cm_test[:, 1]) + np.sum(cm_test[1]) - cm_test[1][1]) if (np.sum(
        cm_test[:, 1]) + np.sum(cm_test[1]) - cm_test[1][1]) != 0 else 0)

    print("---- Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
          " -- Validation: Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(np.sum(cm_test))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          " IoU (TP / (TP + FP + FN))= " + "{:.6f}".format(_sum_iou) +
          " Confusion Matrix= " + np.array_str(cm_test).replace("\n", "")
          )


def train(data, labels, class_distribution, validation_class_distribution, lr_initial, batch_size, niter, weight_decay,
          crop_size, output_path, former_model_path, net_type, has_validation, mean, std):
    channels = len(data[0][1][0][0])
    total_length = len(class_distribution[0]) + len(class_distribution[1])
    total_class_distribution = class_distribution[0] + class_distribution[1]

    display_step = 50
    epoch_number = 10000  # int(len(training_patches)/batch_size) # 1 epoch = images / batch 
    val_inteval = 10000  # int(len(training_patches)/batch_size)

    # Network Parameters
    n_input_data = crop_size * crop_size * channels  # channels
    n_input_data_mask = crop_size * crop_size * 1  # BW
    dropout = 0.5  # Dropout, probability to keep units

    # tf Graph input_data
    x = tf.placeholder(tf.float32, [None, n_input_data])
    y = tf.placeholder(tf.float32, [None, n_input_data_mask])

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # CONVNET
    if net_type == 'deconvnet_1':
        upscore = deconvnet_1(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'deconvnet_2':
        upscore = deconvnet_2(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'dilatedICPR':
        upscore = dilated_convnet_1(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'dilatedGRSL':
        upscore = dilated_convnet_2(x, keep_prob, is_training, weight_decay, crop_size, channels)
    else:
        print BatchColors.FAIL + 'Network type not found: ' + net_type + BatchColors.ENDC
        return

    # Define loss and optimizer
    loss = loss_def(upscore, y)

    global_step = tf.Variable(0, name='dc_global_step', trainable=False)
    lr = tf.train.exponential_decay(lr_initial, global_step, 50000, 0.05, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

    # Evaluate model
    pred_up = tf.argmax(upscore, dimension=3)

    shuffle = np.asarray(random.sample(xrange(3 * total_length), 3 * total_length))
    current_iter = 1

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    saver_restore = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        if 'model' in former_model_path:
            current_iter = int(former_model_path.split('-')[-1])
            print 'Model restored from ' + former_model_path
            saver_restore.restore(sess, former_model_path)
        else:
            sess.run(init)
            print 'Model totally initialized!'
        # saver_restore.restore(sess, output_path + 'model_90000')
        # print BatchColors.OKGREEN + 'Model restored!' + BatchColors.ENDC

        it = 0
        epoch_mean = 0.0
        epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

        # Keep training until reach max iterations
        for step in xrange(current_iter, niter + 1):
            b_x, b_y = dynamically_create_patches(data, labels, crop_size, total_class_distribution,
                                                  shuffle[it * batch_size:min(it * batch_size + batch_size, 3 * total_length)])
            normalize_images(b_x, mean, std)

            batch_x = np.reshape(b_x, (-1, n_input_data))
            batch_y = np.reshape(b_y, (-1, n_input_data_mask))

            # Run optimization op (backprop)
            _, loss_batch, pred_up_batch = sess.run([optimizer, loss, pred_up],
                                                    feed_dict={x: batch_x, y: batch_y, keep_prob: dropout,
                                                               is_training: True})
            acc, batch_cm_train = calc_accuracy_by_crop(batch_y, pred_up_batch, epoch_cm_train)
            epoch_mean += acc

            # BATCH LOSS
            if step != 0 and step % display_step == 0:
                # norm accuracy
                _sum = 0.0
                for i in xrange(len(batch_cm_train)):
                    _sum += (
                        batch_cm_train[i][i] / float(np.sum(batch_cm_train[i])) if np.sum(
                            batch_cm_train[i]) != 0 else 0)

                # IoU accuracy
                _sum_iou = (batch_cm_train[1][1] / float(
                    np.sum(batch_cm_train[:, 1]) + np.sum(batch_cm_train[1]) - batch_cm_train[1][1])
                            if (np.sum(batch_cm_train[:, 1]) + np.sum(batch_cm_train[1]) - batch_cm_train[1][1]) != 0
                            else 0)

                print("Iter " + str(step) +
                      " -- Time " + str(datetime.datetime.now().time()) +
                      " -- Training Minibatch: Loss= " + "{:.6f}".format(loss_batch) +
                      " Overall Accuracy= " + "{:.4f}".format(acc / float(np.sum(batch_cm_train))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                      " IoU (TP / (TP + FP + FN))= " + "{:.4f}".format(_sum_iou) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )

            # EPOCH LOSS
            if step != 0 and step % epoch_number == 0:
                # norm accuracy
                _sum = 0.0
                for i in xrange(len(epoch_cm_train)):
                    _sum += (
                        epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(
                            epoch_cm_train[i]) != 0 else 0)

                # IoU accuracy
                _sum_iou = (epoch_cm_train[1][1] / float(
                    np.sum(epoch_cm_train[:, 1]) + np.sum(epoch_cm_train[1]) - epoch_cm_train[1][1])
                            if (np.sum(epoch_cm_train[:, 1]) + np.sum(epoch_cm_train[1]) - epoch_cm_train[1][1]) != 0
                            else 0)

                print("-- Iter " + str(step) + " -- Training Epoch:" +
                      " -- Time " + str(datetime.datetime.now().time()) +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(np.sum(epoch_cm_train))) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                      " IoU (TP / (TP + FP + FN))= " + "{:.4f}".format(_sum_iou) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            # VALIDATION
            if step != 0 and step % val_inteval == 0:
                # Test
                saver.save(sess, output_path + 'model_' + str(step))
                if has_validation is True:
                    validate(sess, data, labels, validation_class_distribution, n_input_data, n_input_data_mask,
                             batch_size, x, y,
                             keep_prob, is_training, pred_up, upscore, step, crop_size, mean, std, output_path)

            if min(it * batch_size + batch_size,
                   3 * total_length) == 3 * total_length or 3 * total_length == it * batch_size + batch_size:
                # shuffle = np.asarray(random.sample(xrange(total_length), total_length))
                shuffle = np.asarray(random.sample(xrange(3 * total_length), 3 * total_length))
                it = -1
            it += 1

        print("Optimization Finished!")

        # Test: Final
        saver.save(sess, output_path + 'model_final')
        if has_validation is True:
            validate(sess, data, labels, validation_class_distribution, n_input_data, n_input_data_mask, batch_size, x,
                     y, keep_prob,
                     is_training, pred_up, upscore, step, crop_size, mean, std, output_path)
    tf.reset_default_graph()


def validate_from_previous_model(data, labels, validation_class_distribution, batch_size, weight_decay, crop_size,
                                 former_model_path,
                                 output_path, net_type, mean, std):
    channels = len(data[0][1][0][0])

    # TRAIN NETWORK
    # Network Parameters
    n_input_data = crop_size * crop_size * channels  # channels
    n_input_data_mask = crop_size * crop_size * 1  # BW
    dropout = 0.5  # Dropout, probability to keep units

    # tf Graph input_data
    x = tf.placeholder(tf.float32, [None, n_input_data])
    y = tf.placeholder(tf.float32, [None, n_input_data_mask])

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # CONVNET
    if net_type == 'deconvnet_1':
        upscore = deconvnet_1(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'deconvnet_2':
        upscore = deconvnet_2(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'dilatedICPR':
        upscore = dilated_convnet_1(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'dilatedGRSL':
        upscore = dilated_convnet_2(x, keep_prob, is_training, weight_decay, crop_size, channels)
    else:
        print 'Network type not found!'
        return

    # Evaluate model
    pred_up = tf.argmax(upscore, dimension=3)

    saver_restore = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        saver_restore.restore(sess, former_model_path)
        print(BatchColors.OKGREEN + 'Model restored!' + BatchColors.ENDC)

        validate(sess, data, labels, validation_class_distribution, n_input_data, n_input_data_mask, batch_size, x, y,
                 keep_prob,
                 is_training, pred_up, upscore, 0, crop_size, mean, std, output_path)
    tf.reset_default_graph()


def test(data, labels, test_classes_distribution, batch_size, niter, weight_decay, crop_size, former_model_path,
         output_path, net_type, mean, std):
    off = len(data)
    h, w, channels = data[0][1].shape
    print off, h, w, channels

    total_length = len(test_classes_distribution[0]) + len(test_classes_distribution[1])
    total_class_distribution = test_classes_distribution[0] + test_classes_distribution[1]

    # TRAIN NETWORK
    # Network Parameters
    n_input_data = crop_size * crop_size * channels  # channels
    n_input_data_mask = crop_size * crop_size * 1  # BW
    dropout = 0.5  # Dropout, probability to keep units

    # tf Graph input_data
    x = tf.placeholder(tf.float32, [None, n_input_data])
    y = tf.placeholder(tf.float32, [None, n_input_data_mask])

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # CONVNET
    if net_type == 'deconvnet_1':
        upscore = deconvnet_1(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'deconvnet_2':
        upscore = deconvnet_2(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'dilated_convnet_1':
        upscore = dilated_convnet_1(x, keep_prob, is_training, weight_decay, crop_size, channels)
    elif net_type == 'dilated_convnet_2':
        upscore = dilated_convnet_2(x, keep_prob, is_training, weight_decay, crop_size, channels)
    else:
        print 'Network type not found!'
        return

    # Define loss and optimizer
    loss = loss_def(upscore, y)

    # Evaluate model
    pred_up = tf.argmax(upscore, dimension=3)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, former_model_path)
        print BatchColors.OKGREEN + "Model restored: " + former_model_path + BatchColors.ENDC

        listIndex = np.arange(total_length)

        prob_im = np.zeros([off, h, w, NUM_CLASSES], dtype=np.float32)
        occur_im = np.zeros([off, h, w, NUM_CLASSES], dtype=np.float32)

        for i in xrange(0, (
                (total_length / batch_size) if (total_length % batch_size) == 0 else (total_length / batch_size) + 1)):
            batch = listIndex[i * batch_size:min(i * batch_size + batch_size, total_length)]
            b_x, b_y = dynamically_create_patches(data, labels, crop_size, total_class_distribution, batch)
            normalize_images(b_x, mean, std)

            batch_x = np.reshape(b_x, (-1, n_input_data))
            batch_y = np.reshape(b_y, (-1, n_input_data_mask))

            _pred_up, _upscore = sess.run([pred_up, upscore],
                                          feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})
            for j in xrange(len(batch)):
                cur_map = total_class_distribution[batch[j]][0]
                cur_x = total_class_distribution[batch[j]][1][0]
                cur_y = total_class_distribution[batch[j]][1][1]

                prob_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _upscore[j, :, :, :]
                occur_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += 1

        if 0 in occur_im:
            occur_im[np.where(occur_im == 0)] = 1

        prob_map = prob_im / occur_im.astype(float)
        np.save(output_path + 'prob_map.npy', prob_map)
        prob_im_argmax = np.argmax(prob_map, axis=3)
        save_map(output_path, prob_im_argmax, data)

    tf.reset_default_graph()


'''
python segmentation.py /home/mediaeval17/FDSI/ /home/mediaeval17/FDSI/aux/ /home/mediaeval17/FDSI/ 0.01 0.005 100 100 50 25 deconvnet_1 0 training
'''


def main():
    list_params = ['input_data_path', 'output_path(for model, images, etc)', 'former_model_path',
                   'learningRate', 'weight_decay', 'batch_size', 'niter', 'crop_size', 'strideCrop',
                   'networkType (deconvnet_1|deconvnet_2|dilated_convnet_1|dilated_convnet_2)', 'specific_event[0 to all]',
                   'process[training|validate|testing]']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    # input_data path
    index = 1
    input_data_path = sys.argv[index]
    # output path
    index = index + 1
    output_path = sys.argv[index]
    # model path
    index = index + 1
    former_model_path = sys.argv[index]

    # Parameters
    index = index + 1
    lr_initial = float(sys.argv[index])
    index = index + 1
    weight_decay = float(sys.argv[index])
    index = index + 1
    batch_size = int(sys.argv[index])
    index = index + 1
    niter = int(sys.argv[index])

    index = index + 1
    crop_size = int(sys.argv[index])
    index = index + 1
    strideCrop = int(sys.argv[index])
    index = index + 1
    net_type = sys.argv[index]
    index = index + 1
    specific_event = (int(sys.argv[index]) if int(sys.argv[index]) != 0 else None)
    index = index + 1
    process = sys.argv[index]

    has_validation = True

    # PROCESS IMAGES
    # Original Images
    print BatchColors.WARNING + 'Reading Data...' + BatchColors.ENDC
    data, labels = load_images(input_data_path, specific_event=specific_event)
    if process == 'testing':
        print BatchColors.WARNING + 'Reading Test Data...' + BatchColors.ENDC
        test_data, _ = load_images(input_data_path + 'test/', specific_event=specific_event)
        test_fake_labels = []
        for i in xrange(len(test_data)):
            test_fake_labels.append((test_data[i][0], np.zeros((len(test_data[0][1][0]),
                                                                len(test_data[0][1][1])), dtype=np.uint8)))
    elif process == 'testing_diff':
        print BatchColors.WARNING + 'Reading Test Data...' + BatchColors.ENDC
        test_data, _ = load_images(input_data_path + 'test/new_test_diff/', specific_event=specific_event)
        test_fake_labels = []
        for i in xrange(len(test_data)):
            test_fake_labels.append(
                (test_data[i][0], np.zeros((len(test_data[0][1][0]), len(test_data[0][1][1])), dtype=np.uint8)))

    if process == 'training':
        print BatchColors.WARNING + 'Class distribution...' + BatchColors.ENDC

        if specific_event is None:
            class_name = 'classDistribution_' + str(crop_size) + '.npy'
            if has_validation is True:
                validation_name = 'validationClassDistribution_' + str(crop_size) + '.npy'
        else:
            class_name = 'classDistribution_' + str(crop_size) + '_' + str(specific_event) + '.npy'
            if has_validation is True:
                validation_name = 'validationClassDistribution_' + str(crop_size) + '_' + str(specific_event) + '.npy'

        if os.path.isfile(os.getcwd() + '/' + class_name):
            class_distribution = np.load(os.getcwd() + '/' + class_name)
            if has_validation is True:
                validation_class_distribution = np.load(os.getcwd() + '/' + validation_name)
            print(BatchColors.OKGREEN + '...loaded!' + BatchColors.ENDC)
        else:
            class_distribution = create_distributions_over_classes(labels, crop_size, strideCrop)
            np.save(os.getcwd() + '/' + class_name, class_distribution)
            if has_validation is True:
                validation_class_distribution = create_balanced_validation_set(class_distribution, percentage_val=0.2)
                np.save(os.getcwd() + '/' + validation_name, validation_class_distribution)
            print(BatchColors.OKGREEN + '...created!' + BatchColors.ENDC)

        for i in xrange(len(class_distribution)):
            print BatchColors.OKBLUE + 'Class ' + str(i + 1) + ' has ' + str(len(class_distribution[i])) + '/' + str(
                len(validation_class_distribution[i])) + ' instances' + BatchColors.ENDC

        mean, std = dynamically_calculate_mean_and_std(data, crop_size, class_distribution)

        nets = ['segNet', 'segnetICPR', 'dilatedGRSL']  # ['segNet', 'segnetICPR', 'dilatedICPR', 'dilatedGRSL']
        # 'deconvICPR', 'deconv25']
        for net in nets:
            if specific_event is None:
                currentoutput_path = output_path + 'output_' + net + '_' + str(crop_size) + '/'
            else:
                currentoutput_path = output_path + 'output_' + net + '_' + str(crop_size) + '_' + str(
                    specific_event) + '/'
            if not os.path.isdir(currentoutput_path):
                os.makedirs(currentoutput_path)

            print(BatchColors.OKGREEN + 'Current network: ' + net + BatchColors.ENDC)
            if has_validation is True:
                train(data, labels, class_distribution, validation_class_distribution, lr_initial, batch_size, niter,
                      weight_decay, crop_size, output_path=currentoutput_path, former_model_path=former_model_path,
                      net_type=net, has_validation=has_validation, mean=mean, std=std)
            else:
                train(data, labels, class_distribution, [], lr_initial, batch_size, niter, weight_decay, crop_size,
                      output_path=currentoutput_path, former_model_path=former_model_path,
                      net_type=net, has_validation=has_validation, mean=mean, std=std)
    elif process == 'validate':
        if specific_event is None:
            class_name = 'classDistribution_' + str(crop_size) + '.npy'
            validation_name = 'validationClassDistribution_' + str(crop_size) + '.npy'
        else:
            class_name = 'classDistribution_' + str(crop_size) + '_' + str(specific_event) + '.npy'
            validation_name = 'validationClassDistribution_' + str(crop_size) + '_' + str(specific_event) + '.npy'

        if os.path.isfile(os.getcwd() + '/' + class_name):
            class_distribution = np.load(os.getcwd() + '/' + class_name)
            validation_class_distribution = np.load(os.getcwd() + '/' + validation_name)
            print(BatchColors.OKGREEN + 'Distribution Loaded!' + BatchColors.ENDC)
        else:
            class_distribution = create_distributions_over_classes(labels, crop_size, strideCrop)
            validation_class_distribution = create_balanced_validation_set(class_distribution, percentage_val=0.2)
            np.save(os.getcwd() + '/' + validation_name, validation_class_distribution)
            print(BatchColors.OKGREEN + 'Distribution Saved!' + BatchColors.ENDC)
        mean, std = dynamically_calculate_mean_and_std(data, crop_size, class_distribution)

        # exclude others
        # specific_event = 1
        # range = [143,110,137,18,31,23]
        # rangeMin = 0
        # rangeMax = 0
        # for i in xrange(len(range)):
        # if i < specific_event-1:
        # rangeMin += range[i]
        # rangeMax = rangeMin + range[specific_event-1]
        # print rangeMin, rangeMax

        # validation_class_distribution = validation_class_distribution[0] + validation_class_distribution[1]
        # print len(validation_class_distribution)
        # validation_class_distribution = [i for i in validation_class_distribution if i[0] >= rangeMin and i[0] < rangeMax]
        # print len(validation_class_distribution)

        validate_from_previous_model(data, labels, validation_class_distribution, batch_size, weight_decay, crop_size,
                                     former_model_path, output_path, net_type, mean=mean, std=std)
    elif 'testing' in process:
        # TEST
        if specific_event is None:
            class_name = 'classDistribution_' + str(crop_size) + '.npy'
        else:
            class_name = 'classDistribution_' + str(crop_size) + '_' + str(specific_event) + '.npy'

        if os.path.isfile(os.getcwd() + '/' + class_name):
            class_distribution = np.load(os.getcwd() + '/' + class_name)
            print(BatchColors.OKGREEN + 'Distribution Loaded!' + BatchColors.ENDC)
        else:
            class_distribution = create_distributions_over_classes(labels, crop_size, strideCrop)
            print(BatchColors.OKGREEN + 'Distribution Saved!' + BatchColors.ENDC)
        mean, std = dynamically_calculate_mean_and_std(data, crop_size, class_distribution)

        testclass_distribution = create_distributions_over_classes(test_fake_labels, crop_size, strideCrop)
        test(test_data, test_fake_labels, testclass_distribution, batch_size, niter, weight_decay, crop_size,
             former_model_path, output_path, net_type, mean, std)
    else:
        print BatchColors.FAIL + 'Process ' + process + \
              ' not found! Options are: training, validate and testing.' + BatchColors.ENDC


if __name__ == "__main__":
    main()
