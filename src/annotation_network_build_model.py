
# --------------------------------------------------
#
#     Copyright (C) {2020}  {Kevin Bronik and Le Zhang}
#
#     UCL Medical Physics and Biomedical Engineering
#     https://www.ucl.ac.uk/medical-physics-biomedical-engineering/
#     UCL Queen Square Institute of Neurology
#     https://www.ucl.ac.uk/ion/

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     {Multi-Label Multi/Single-Class Image Segmentation}  Copyright (C) {2020}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.

# This program uses piece of source code from:
# Title: nicMSlesions
# Author: Sergi Valverde
# Date: 2017
# Code version: 0.2
# Availability: https://github.com/NIC-VICOROB/nicMSlesions

# --------------------------------------------------


import os
import signal
import time
import shutil
import numpy as np
import threading
import Okeras
from Okeras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from .annotation_network_build_nets import build_annotation_network
from Okeras import optimizers, losses
import tensorflow as tf

from numpy import inf
from Okeras import backend as K
from scipy.spatial.distance import directed_hausdorff, chebyshev

# force data format to "channels first"
Okeras.backend.set_image_data_format('channels_first')

#################### make the data generator threadsafe ####################

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g



def data_augmentation(Xb, yb):

    # Flip a given percentage of the images at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs // 2, replace=False)
    x_da = Xb[indices]

    # apply rotation to the input batch
    rotate_90 = x_da[:, :, :, ::-1, :].transpose(0, 1, 2, 4, 3)
    rotate_180 = rotate_90[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)

    # apply flipped versions of rotated patches
    rotate_0_flipped = x_da[:, :, :, :, ::-1]
    rotate_180_flipped = rotate_180[:, :, :, :, ::-1]

    augmented_x = np.stack([rotate_180,
                            rotate_0_flipped,
                            rotate_180_flipped],
                            axis=1)

    # select random indices from computed data_augmentationations
    r_indices = np.random.randint(0, 3, size=augmented_x.shape[0])

    Xb[indices] = np.stack([augmented_x[i,
                                        r_indices[i], :, :, :, :]
                            for i in range(augmented_x.shape[0])])

    return Xb, yb


# def train_data_generator(x_train, y_train, batch_size=256):
#     """
#     Keras generator used for training with data augmentation. This generator
#     calls the data augmentation function yielding training samples
#     """
#     num_samples = x_train.shape[0]
#     while True:
#         for b in range(0, num_samples, batch_size):
#             x_ = x_train[b:b+batch_size]
#             y_ = y_train[b:b+batch_size]
#             x_, y_ = data_augmentation(x_, y_)
#             yield x_, y_
@threadsafe_generator
def train_data_generator(x_train, y_train, numsample, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """

    num_samples = int(numsample / batch_size) * batch_size


    # y_all = {}
    # x_all = {}
    while True:
        # for i in range (0,12):
        for b in range(0, num_samples, batch_size):
            # x_ = x_train[b:b + batch_size]
            x_ = [x_train[i][b:b + batch_size] for i in range(0, 5)]
            y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # x_, y_ = data_augmentation([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            for i in range(0,5):
                x_[i], y_[i] = data_augmentation(x_[i], y_[i])

            # yield x_, y_
            yield ([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            # yield (x_all[0], [y_all[i] for i in range (0,12)])

@threadsafe_generator
def cross_vld_data_generator(x_train_v, y_train_v, numsample, batch_size=256):


    num_samples = int(numsample / batch_size) * batch_size

    # y_all = {}
    # x_all = {}
    while True:
        # for i in range (0,12):
        for b in range(0, num_samples, batch_size):
            # x_ = x_train[b:b + batch_size]
            x_ = [x_train_v[i][b:b + batch_size] for i in range(0, 5)]
            y_ = [y_train_v[i][b:b + batch_size] for i in range(0, 5)]
            # y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # x_, y_ = data_augmentation([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            for i in range(0, 5):
                x_[i], y_[i] = data_augmentation(x_[i], y_[i])

            # yield x_, y_
            yield ([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            # yield (x_all[0], [y_all[i] for i in range (0,12)])

@threadsafe_generator
def train_data_generator_multi_class(x_train, y_train, numsample, n_dim, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """

    num_samples = int(numsample / batch_size) * batch_size


    # y_all = {}
    # x_all = {}
    while True:
        # for i in range (0,12):
        for b in range(0, num_samples, batch_size):
            # x_ = x_train[b:b + batch_size]
            x_ = [x_train[i][b:b + batch_size] for i in range(0, n_dim)]
            y_ = [y_train[i][b:b + batch_size] for i in range(0, n_dim)]
            # y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # x_, y_ = data_augmentation([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            for i in range(0,n_dim):
                x_[i], y_[i] = data_augmentation(x_[i], y_[i])

            # yield x_, y_
            yield ([x_[i] for i in range(0, n_dim)], [y_[i] for i in range(0, n_dim)])
            # yield (x_all[0], [y_all[i] for i in range (0,12)])

@threadsafe_generator
def cross_vld_data_generator_multi_class(x_train_v, y_train_v, numsample, n_dim, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """

    num_samples = int(numsample / batch_size) * batch_size

    # y_all = {}
    # x_all = {}
    while True:
        # for i in range (0,12):
        for b in range(0, num_samples, batch_size):
            # x_ = x_train[b:b + batch_size]
            x_ = [x_train_v[i][b:b + batch_size] for i in range(0, n_dim)]
            y_ = [y_train_v[i][b:b + batch_size] for i in range(0, n_dim)]
            # y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # x_, y_ = data_augmentation([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            for i in range(0, n_dim):
                x_[i], y_[i] = data_augmentation(x_[i], y_[i])

            # yield x_, y_
            yield ([x_[i] for i in range(0, n_dim)], [y_[i] for i in range(0, n_dim)])
            # yield (x_all[0], [y_all[i] for i in range (0,12)])

def rand_bin_array(K, N):
        arr = np.zeros(N)
        arr[:K] = 1
        # np.random.shuffle(arr)
        return arr

def _to_tensor(x, dtype):
        x = tf.convert_to_tensor(x)
        if x.dtype != dtype:
            x = tf.cast(x, dtype)
        return x

def Jaccard_index(y_true, y_pred):
        smooth = 100.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        score = (intersection + smooth) / (union + smooth)
        return score

def Jaccard_loss(y_true, y_pred):
        loss = 1 - Jaccard_index(y_true, y_pred)
        return loss

def true_false_positive_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):

        # y_pred_f = tf.reshape(y_pred, [-1])

        # y_pred_f = tf.reshape(y_pred, [-1])
        # this_size = K.int_shape(y_pred_f)[0]
        # # arr_len = this_size
        # # num_ones = 1
        # # arr = np.zeros(arr_len, dtype=int)
        # # if this_size is not None:
        # #      num_ones= np.int32((this_size * value * 100) / 100)
        # #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
        # #      arr[idx] = 1
        # #      p_labels = arr
        # # else:
        # #      p_labels = np.random.randint(2, size=this_size)
        # if this_size is not None:
        #      p_labels = np.random.binomial(1, value, size=this_size)
        #      p_labels = tf.reshape(p_labels, y_pred.get_shape())
        # else:
        #      p_labels =  y_pred

        # p_lprint ('num_classes .....///', num_classes.shape[0])abels = tf.constant(y_pred_f)
        # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
        # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)

        # num_classes = 50
        # print ('num_classes .....', num_classes)
        # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
        # p_labels = tf.reduce_max(p_labels, 0)
        # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
        # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
        # y_true = K.cast(y_true, y_pred.dtype)

        if label_smoothing is not 0:
            smoothing = K.cast_to_floatx(label_smoothing)

            def _smooth_labels():
                num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
                return y_true * (1.0 - smoothing) + (smoothing / num_classes)

            y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

        C11 = tf.math.multiply(y_true, y_pred)
        c_y_pred = 1 - y_pred
        C12 = tf.math.multiply(y_true, c_y_pred)
        weighted_y_pred_u = tf.math.multiply(K.cast(p_labels, y_pred.dtype), y_pred)
        weighted_y_pred_d = 1 - weighted_y_pred_u
        y_pred = tf.math.add(tf.math.multiply(C11, weighted_y_pred_u), tf.math.multiply(C12, weighted_y_pred_d))

        # y_pred /= tf.reduce_sum(y_pred,
        #                             reduction_indices=len(y_pred.get_shape()) - 1,
        #                             keep_dims=True)
        #     # manual computation of crossentropy
        # _EPSILON = 10e-8
        # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
        #                            reduction_indices=len(y_pred.get_shape()) - 1)

        loss = Jaccard_loss(y_true, y_pred)
        # with tf.GradientTape() as t:
        #     t.watch(y_pred)
        #     dpred = t.gradient(loss, y_pred)

        return loss

        # return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

def false_true_negative_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):

        # arr_len = this_size
        # num_ones = 1
        # arr = np.zeros(arr_len, dtype=int)
        # if this_size is not None:
        #      num_ones= np.int32((this_size * value * 100) / 100)
        #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
        #      arr[idx] = 1
        #      p_labels = arr
        # else:
        #      p_labels = np.random.randint(2, size=this_size)
        # np.random.binomial(1, 0.34, size=10)

        # if this_size is not None:
        #     this_value= np.int32((this_size * value * 100) / 100)
        # else:
        #     this_value =  1

        # p_labels = np.random.randint(2, size=this_size)

        # p_labels = tf.constant(y_pred_f)
        # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
        # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)
        # num_classes = 50
        # print ('num_classes .....', num_classes)
        # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
        # # p_labels = tf.reduce_max(p_labels, 0)
        # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
        # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
        # y_true = K.cast(y_true, y_pred.dtype)

        if label_smoothing is not 0:
            smoothing = K.cast_to_floatx(label_smoothing)

            def _smooth_labels():
                num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
                return y_true * (1.0 - smoothing) + (smoothing / num_classes)

            y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

        c_y_true = 1 - y_true
        c_y_pred = 1 - y_pred
        C21 = tf.math.multiply(c_y_true, y_pred)

        C22 = tf.math.multiply(c_y_true, c_y_pred)
        weighted_y_pred_u = tf.math.multiply(K.cast(p_labels, y_pred.dtype), y_pred)
        weighted_y_pred_d = 1 - weighted_y_pred_u

        y_pred = tf.math.add(tf.math.multiply(C21, weighted_y_pred_u), tf.math.multiply(C22, weighted_y_pred_d))
        # y_pred /= tf.reduce_sum(y_pred,
        #                             reduction_indices=len(y_pred.get_shape()) - 1,
        #                             keep_dims=True)
        #     # manual computation of crossentropy
        # _EPSILON = 10e-8
        # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
        #                            reduction_indices=len(y_pred.get_shape()) - 1)
        # with tf.GradientTape() as t:
        #     t.watch(y_pred)
        #     dpred = t.gradient(loss, y_pred)
        y_true = 1 - y_true
        loss = Jaccard_loss(y_true, y_pred)

        return loss

def penalty_loss_trace_normalized_confusion_matrix(y_true, y_pred):
        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred

        tp = K.sum(y_true * y_pred)
        fn = K.sum(y_true * neg_y_pred)
        sum1 = tp + fn

        fp = K.sum(neg_y_true * y_pred)
        tn = K.sum(neg_y_true * neg_y_pred)
        sum2 = fp + tn

        tp_n = tp / sum1
        fn_n = fn / sum1
        fp_n = fp / sum2
        tn_n = tn / sum2
        trace = (tf.math.square(tp_n) + tf.math.square(tn_n) + tf.math.square(fn_n) + tf.math.square(fp_n))

        return (1 - trace * 0.5) / 5

def p_loss(y_true, y_pred):
        smooth = 100.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        pg1 = (2. * (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)) * tf.reduce_sum(y_true_f)
        pg2 = (2. * intersection + smooth)
        pg3 = K.square(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        pg = (pg1 - pg2) / pg3

        # w = tf.Variable(y_pred_f, trainable=True)
        # with tf.GradientTape() as t:
        #      t.watch(y_pred_f)
        #
        # pg = t.gradient(score, y_pred_f)

        # pg = K.gradients(score, y_pred_f)[0]
        # pg = K.sqrt(sum([K.sum(K.square(g)) for g in pg_t]))

        return score, pg

def constrain(y_true, y_pred):
        loss, g = p_loss(y_true, y_pred)
        return loss

def constrain_loss(y_true, y_pred):
        return 1 - constrain(y_true, y_pred)

    # def augmented_Lagrangian_loss(y_true, y_pred, augment_Lagrangian=1):

    #     C_value, pgrad = p_loss(y_true, y_pred)
    #     Ld, grad1 = loss_down(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
    #     Lu, grad2 = loss_up(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
    #     ploss = 1 - C_value
    #     # adaptive lagrange multiplier
    #     _EPSILON = 10e-8
    #     if all(v is not None for v in [grad1, grad2, pgrad]):
    #          alm = - ((grad1 + grad2) / pgrad + _EPSILON)
    #     else:
    #          alm =  augment_Lagrangian
    #     ploss = ploss * alm
    #     total_loss = Ld + Lu + ploss
    #     return total_loss
def calculate_gradient(y_true, y_pred):
    constrain_l = constrain_loss(y_true, y_pred)
    this_value = (-1 * constrain_l) + 1
    y_pred_f = tf.reshape(y_pred, [-1])
    this_size = K.int_shape(y_pred_f)[0]
    if this_size is not None:
        #  numpy.random.rand(4)
        #  p_labels = np.random.binomial(1, this_value, size=this_size)
        p_labels_g = rand_bin_array(this_value, this_size)
        p_labels_g = my_func(np.array(p_labels_g, dtype=np.float32))
        # p_labels = tf.dtypes.cast((tf.dtypes.cast(tf.size(p_labels), tf.int32)
        p_labels_g = tf.reshape(p_labels_g, y_pred.get_shape())

    else:
        p_labels_g = 0

    loss1 = true_false_positive_loss(y_true, y_pred, p_labels=p_labels_g, label_smoothing=0, value=this_value)
    loss2 = false_true_negative_loss(y_true, y_pred, p_labels=p_labels_g, label_smoothing=0, value=this_value)

    g_loss1 = K.sum(K.gradients(loss1, y_pred)[0])
    g_loss2 = K.sum(K.gradients(loss2, y_pred)[0])
    # cg_loss = K.sum(K.gradients(losses.categorical_crossentropy(y_true, y_pred), y_pred)[0])
    # pg_loss = K.sum(K.gradients(penalty_loss_trace_normalized_confusion_matrix(y_true, y_pred), y_pred)[0])

    return g_loss1, g_loss2



#
# def calculate_gradient(y_true, y_pred, loss1, loss2):
#
#         # w = tf.Variable(y_pred,  trainable=True)
#         # with tf.GradientTape(persistent=True) as t:
#         #       t.watch(y_pred)
#         #
#         # g_loss1 = t.gradient(loss1, y_pred)
#         # g_loss2 = t.gradient(loss2, y_pred)
#
#         g_loss1 = K.sum(K.gradients(loss1, y_pred)[0])
#         g_loss2 =K.sum(K.gradients(loss2, y_pred)[0])
#         # g_loss1 = K.sqrt(sum([K.sum(K.square(g)) for g in g_loss1_t]))
#         # g_loss2 = K.sqrt(sum([K.sum(K.square(g)) for g in g_loss2_t]))
#         # loss, g_constrain = p_loss (y_true, y_pred)
#         loss, g_constrain = p_loss(y_true, y_pred)
#         return g_loss1, g_loss2, g_constrain

def my_func(arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return tf.matmul(arg, arg) + arg

def Adaptive_Lagrange_Multiplier(y_pred, y_true):

        smooth = 1.

        augment_Lagrangian = 1
        # if all(v is not None for v in [Gloss1, Gloss2, Gloss3]):
        if y_pred is not None:

            loss, g_constrain = p_loss(y_true, y_pred)
            g_loss1, g_loss2 = calculate_gradient(y_true, y_pred)


            res = (g_loss1 + g_loss2 + smooth) / (g_constrain + smooth)
            # res = ((K.sum(Gloss1) + K.sum(Gloss2)))

            # losses.categorical_crossentropy(y_true,
              #                              y_pred) + penalty_loss_trace_normalized_confusion_matrix(
               # y_true, y_pred)


            return res * (K.sum(y_true * y_true) / K.sum(y_true * y_true))
        else:
            # print("adaptive_lagrange_multiplier", augment_Lagrangian)
            return K.sum(y_true * y_true) / K.sum(y_true * y_true)

def Individual_loss(y_true, y_pred):
        # C_value = p_loss(y_true, y_pred)
        constrain_l = constrain_loss(y_true, y_pred)
        this_value = (-1 * constrain_l) + 1
        y_pred_f = tf.reshape(y_pred, [-1])
        this_size = K.int_shape(y_pred_f)[0]
        if this_size is not None:
            #  numpy.random.rand(4)
            #  p_labels = np.random.binomial(1, this_value, size=this_size)
            p_labels = rand_bin_array(this_value, this_size)
            p_labels = my_func(np.array(p_labels, dtype=np.float32))
            # p_labels = tf.dtypes.cast((tf.dtypes.cast(tf.size(p_labels), tf.int32)
            p_labels = tf.reshape(p_labels, y_pred.get_shape())

        else:
            p_labels = 0

        loss1 = true_false_positive_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
        loss2 = false_true_negative_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
        # grad1, grad2, pgrad = calculate_gradient(y_true, y_pred, loss1, loss2)

        # ploss = 1 - C_value
        # adaptive lagrange multiplier
        # adaptive_lagrange_multiplier(y_true, y_pred):
        # _EPSILON = 10e-8
        # if all(v is not None for v in [grad1, grad2, pgrad]):
        #     return (((grad1 + grad2) + _EPSILON) / pgrad + _EPSILON)
        # else:
        #     return  augment_Lagrangian
        # ploss = ploss * alm
        lm = Adaptive_Lagrange_Multiplier(y_pred, y_true)

        # outF = open(
        #     os.path.join('/home/kbronik/Desktop/IE_MULTIINPUT/CNN_GUI_multiinputs_singleoutput_modified_Keras/utils/this.txt'),
        #     "w")
        # parsed_line = str(lm)
        # outF.writelines(parsed_line)
        # outF.close()

        to_loss = loss1 + loss2 + (lm * constrain_l)
        return to_loss + losses.categorical_crossentropy(y_true,
                                                         y_pred) + penalty_loss_trace_normalized_confusion_matrix(
            y_true, y_pred)

    #     return loss
def Total_accuracy(y_true, y_pred):
        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred
        fp = K.sum(neg_y_true * y_pred)
        tn = K.sum(neg_y_true * neg_y_pred)
        fn = K.sum(y_true * neg_y_pred)
        tp = K.sum(y_true * y_pred)
        acc = (tp + tn) / (tp + tn + fn + fp)
        return acc


def True_Positive(y_true, y_pred):
    tp = K.sum(y_true * y_pred)
    return tp

def False_Positive(y_true, y_pred):
    neg_y_true = 1 - y_true
    # neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    return fp

def False_Negative(y_true, y_pred):
    #neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fn = K.sum(y_true * neg_y_pred)
    return fn

def True_Negative(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tn = K.sum(neg_y_true * neg_y_pred)
    return tn



def build_and_compile_models(settings):


    # save model to disk to re-use it. Create an model_name folder
    # organize model_name
    if not os.path.exists(os.path.join(settings['weight_paths'],
                                       settings['model_name'])):
        os.mkdir(os.path.join(settings['weight_paths'],
                              settings['model_name']))
    if not os.path.exists(os.path.join(settings['weight_paths'],
                                       settings['model_name'], 'nets')):
        os.mkdir(os.path.join(settings['weight_paths'],
                              settings['model_name'], 'nets'))
    if settings['debug']:
        if not os.path.exists(os.path.join(settings['weight_paths'],
                                           settings['model_name'],
                                           '.train')):
            os.mkdir(os.path.join(settings['weight_paths'],
                                  settings['model_name'],
                                  '.train'))

    first = 'First'
    second = 'Second'

    this_size = len(settings['all_isolated_label'])

    if this_size == 5:
        model = build_annotation_network(settings, first)
        all_loss = []
        for i in range(0, int(this_size)):
            all_loss.append(Individual_loss)

        model.compile(loss=all_loss,
                      optimizer='adadelta', option=int(this_size),
                      metrics=["accuracy"])
        # model.compile(loss=Individual_loss,
        #               optimizer='adadelta',
        #               metrics=[Total_accuracy, Individual_loss, Adaptive_Lagrange_Multiplier, True_Positive,
        #                        False_Positive, False_Negative, True_Negative, "accuracy"])

        # save weights
        net_model_1 = 'model_1'
        net_weights_1 = os.path.join(settings['weight_paths'],
                                     settings['model_name'],
                                     'nets', net_model_1 + '.hdf5')

        net1 = {}
        net1['net'] = model
        net1['weights'] = net_weights_1
        net1['history'] = None
        net1['special_name_1'] = net_model_1

        # --------------------------------------------------
        # model 2
        # --------------------------------------------------

        model2 = build_annotation_network(settings, second)
        model2.compile(loss=all_loss,
                       optimizer='adadelta', option=int(this_size),
                       metrics=["accuracy"])
        # model2.compile(loss=Individual_loss,
        #                optimizer='adadelta', option=int(this_size / 5),
        #                metrics=[Total_accuracy, Individual_loss, Adaptive_Lagrange_Multiplier, True_Positive,
        #                         False_Positive, False_Negative, True_Negative, "accuracy"])

        net_model_2 = 'model_2'
        net_weights_2 = os.path.join(settings['weight_paths'],
                                     settings['model_name'],
                                     'nets', net_model_2 + '.hdf5')

        net2 = {}
        net2['net'] = model2
        net2['weights'] = net_weights_2
        net2['history'] = None
        net2['special_name_2'] = net_model_2

        # load predefined weights if transfer learning is selected

        if settings['full_train'] is False:

            # load default weights
            print("> CNN_GUI: Loading pretrained weights from the", settings['pretrained_model'], "configuration")
            pretrained_model = os.path.join(settings['weight_paths'], settings['pretrained_model'], 'nets')
            model = os.path.join(settings['weight_paths'],
                                 settings['model_name'])
            net1_w_def = os.path.join(model, 'nets', 'model_1.hdf5')
            net2_w_def = os.path.join(model, 'nets', 'model_2.hdf5')

            if not os.path.exists(model):
                shutil.copy(pretrained_model, model)
            else:
                shutil.copyfile(os.path.join(pretrained_model,
                                             'model_1.hdf5'),
                                net1_w_def)
                shutil.copyfile(os.path.join(pretrained_model,
                                             'model_2.hdf5'),
                                net2_w_def)
            try:
                net1['net'].load_weights(net1_w_def, by_name=True)
                net2['net'].load_weights(net2_w_def, by_name=True)
            except:
                print("> ERROR: The model", settings['model_name'],
                      'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if settings['load_weights'] is True:
            print("> CNN_GUI: loading weights from",
                  settings['model_name'], 'configuration')
            print(net_weights_1)
            print(net_weights_2)

            net1['net'].load_weights(net_weights_1, by_name=True)
            net2['net'].load_weights(net_weights_2, by_name=True)

        return [net1, net2]
    else:
        model = build_annotation_network(settings, first)

        # adaptive deep learning network architecture
        all_loss = []
        for i in range(0, int(this_size)):
            all_loss.append(Individual_loss)

        model.compile(loss=all_loss,
                      optimizer='adadelta', option=int(this_size),
                      metrics=["accuracy"])

        # save weights
        net_model_1 = 'model_1'
        net_weights_1 = os.path.join(settings['weight_paths'],
                                     settings['model_name'],
                                     'nets', net_model_1 + '.hdf5')

        net1 = {}
        net1['net'] = model
        net1['weights'] = net_weights_1
        net1['history'] = None
        net1['special_name_1'] = net_model_1

        # --------------------------------------------------
        # model 2
        # --------------------------------------------------

        model2 = build_annotation_network(settings, second)
        model2.compile(loss=all_loss,
                       optimizer='adadelta', option=int(this_size),
                       metrics=["accuracy"])

        net_model_2 = 'model_2'
        net_weights_2 = os.path.join(settings['weight_paths'],
                                     settings['model_name'],
                                     'nets', net_model_2 + '.hdf5')

        net2 = {}
        net2['net'] = model2
        net2['weights'] = net_weights_2
        net2['history'] = None
        net2['special_name_2'] = net_model_2

        # load predefined weights if transfer learning is selected

        if settings['full_train'] is False:

            # load default weights
            print("> CNN_GUI: Loading pretrained weights from the",  settings['pretrained_model'], "configuration")
            pretrained_model = os.path.join(settings['weight_paths'], settings['pretrained_model'], 'nets')
            model = os.path.join(settings['weight_paths'],
                                 settings['model_name'])
            net1_w_def = os.path.join(model, 'nets', 'model_1.hdf5')
            net2_w_def = os.path.join(model, 'nets', 'model_2.hdf5')

            if not os.path.exists(model):
                shutil.copy(pretrained_model, model)
            else:
                shutil.copyfile(os.path.join(pretrained_model,
                                             'model_1.hdf5'),
                                net1_w_def)
                shutil.copyfile(os.path.join(pretrained_model,
                                             'model_2.hdf5'),
                                net2_w_def)
            try:
                net1['net'].load_weights(net1_w_def, by_name=True)
                net2['net'].load_weights(net2_w_def, by_name=True)
            except:
                print("> ERROR: The model", settings['model_name'], 'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if settings['load_weights'] is True:
            print("> CNN_GUI: loading weights from", settings['model_name'], 'configuration')
            print(net_weights_1)
            print(net_weights_2)

            net1['net'].load_weights(net_weights_1, by_name=True)
            net2['net'].load_weights(net_weights_2, by_name=True)

        return [net1, net2]



def redefine_network_layers_for_training(model, settings, num_layers=1, number_of_samples=None):

    if number_of_samples is not None:
        if number_of_samples < 10000:
            num_layers = 1
        elif number_of_samples < 100000:
            num_layers = 2
        else:
            num_layers = 3

    # all layers are first set to non trainable

    net = model['net']
    for l in net.layers:
         l.trainable = False

    print("> CNN_GUI: re-training the last", num_layers, "layers")


    this_size = len(settings)
    if this_size > 5:
        for i in range(0, int(this_size)):
            this_name = 'output' + str(i + 1)
            net.get_layer(this_name).trainable = True
    else:
         net.get_layer('output1').trainable = True
         net.get_layer('output2').trainable = True
         net.get_layer('output3').trainable = True
         net.get_layer('output4').trainable = True
         net.get_layer('output5').trainable = True
    # re-train the FC layers based on the number of retrained
    # layers
    net.get_layer('out').trainable = True

    if num_layers == 1:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
    if num_layers == 2:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
    if num_layers == 3:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
        net.get_layer('dr_d1').trainable = True
        net.get_layer('d1').trainable = True
        net.get_layer('prelu_d1').trainable = True

    #net.compile(loss='categorical_crossentropy',
    #            optimizer='adadelta',
    #            metrics=['accuracy'])

    model['net'] = net
    return model
def calcul_num_sample(x_train):
    temp = {}
    for label in range(0,5):
         temp[label] = x_train[label].shape[0]
         # print("temp[label]",label, temp[label])

    num_samples_t = min(temp.items(), key=lambda x: x[1])
    num_samples = num_samples_t[1]
    return num_samples

def calcul_num_sample_multi_class(x_train, label_n):
    temp = {}
    for label in range(0, label_n):
         temp[label] = x_train[label].shape[0]
         # print("temp[label]",label, temp[label])

    num_samples_t = min(temp.items(), key=lambda x: x[1])
    num_samples = num_samples_t[1]
    return num_samples

def fit_multifunctional_model(model, x_train, y_train, settings, X_val, Y_val, initial_epoch=0):

    this_size = len(settings['all_isolated_label'])
    num_epochs = settings['max_epochs']
    train_split_perc = settings['train_split']
    batch_size = settings['batch_size']

    # convert labels to categorical
    # y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
    if this_size == 5:
        for label in range(0, 5):
            y_train[label] = Okeras.utils.to_categorical(y_train[label] == 1,
                                                         len(np.unique(y_train[label] == 1)))
            Y_val[label] = Okeras.utils.to_categorical(Y_val[label] == 1,
                                                       len(np.unique(Y_val[label] == 1)))

        train_val = {}
        train_val_v = {}
        # split training and validation
        for label in range(0, 5):
            perm_indices = np.random.permutation(x_train[label].shape[0])
            train_val[label] = int(len(perm_indices) * train_split_perc)
            perm_indices_v = np.random.permutation(X_val[label].shape[0])
            train_val_v[label] = int(len(perm_indices_v) * train_split_perc)

        x_train_ = {}
        y_train_ = {}
        x_val_ = {}
        y_val_ = {}

        train_val = min(train_val.items(), key=lambda x: x[1])
        train_val = train_val[1]
        train_val_v = min(train_val_v.items(), key=lambda x: x[1])
        train_val_v = train_val_v[1]

        # this = calcul_num_sample(x_train)
        # that = calcul_num_sample(X_val)
        #
        for i in range(0, 5):
            x_train_[i] = x_train[i][:train_val]
            print("x_train_[", i, "]:", x_train_[i].shape[0])
            y_train_[i] = y_train[i][:train_val]
            print("y_train_[", i, "]:", y_train_[i].shape[0])
            x_val_[i] = X_val[i][:train_val_v]
            print("x_val_", i, "]:", x_val_[i].shape[0])
            y_val_[i] = Y_val[i][:train_val_v]
            print("y_val_[", i, "]:", y_val_[i].shape[0])

        numsamplex = calcul_num_sample(x_train_)
        numsamplexv = calcul_num_sample(x_val_)

        print("Number of sample for training:", numsamplex)
        print("Number of sample for cross validation:", numsamplexv)

        history = model['net'].fit_generator(train_data_generator(
            x_train_, y_train_, numsamplex,
            batch_size=batch_size),
            validation_data=cross_vld_data_generator(x_val_, y_val_, numsamplexv, batch_size=batch_size),
            # validation_data = (x_val_, y_val_),
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=int(numsamplex / batch_size),
            validation_steps=int(numsamplexv / batch_size),

            verbose=settings['net_verbose'],
            callbacks=[ModelCheckpoint(model['weights'],
                                       # monitor=['val_all_loss'],
                                       save_best_only=True,
                                       save_weights_only=True),
                       EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=settings['patience'],
                                     mode='auto'), TensorBoard(log_dir='./tensorboardlogs',
                                                               histogram_freq=0,
                                                               write_graph=True)])

    else:
        for label in range(0, this_size):
            y_train[label] = Okeras.utils.to_categorical(y_train[label] == 1,
                                                         len(np.unique(y_train[label] == 1)))
            Y_val[label] = Okeras.utils.to_categorical(Y_val[label] == 1,
                                                       len(np.unique(Y_val[label] == 1)))

        train_val = {}
        train_val_v = {}
        # split training and validation
        for label in range(0, this_size):
            perm_indices = np.random.permutation(x_train[label].shape[0])
            train_val[label] = int(len(perm_indices) * train_split_perc)
            perm_indices_v = np.random.permutation(X_val[label].shape[0])
            train_val_v[label] = int(len(perm_indices_v) * train_split_perc)

        x_train_ = {}
        y_train_ = {}
        x_val_ = {}
        y_val_ = {}

        train_val = min(train_val.items(), key=lambda x: x[1])
        train_val = train_val[1]
        train_val_v = min(train_val_v.items(), key=lambda x: x[1])
        train_val_v = train_val_v[1]

        # this = calcul_num_sample(x_train)
        # that = calcul_num_sample(X_val)
        #
        for i in range(0, this_size):
            x_train_[i] = x_train[i][:train_val]
            print("x_train_[", i, "]:", x_train_[i].shape[0])
            y_train_[i] = y_train[i][:train_val]
            print("y_train_[", i, "]:", y_train_[i].shape[0])
            x_val_[i] = X_val[i][:train_val_v]
            print("x_val_", i, "]:", x_val_[i].shape[0])
            y_val_[i] = Y_val[i][:train_val_v]
            print("y_val_[", i, "]:", y_val_[i].shape[0])

        numsamplex = calcul_num_sample_multi_class(x_train_, this_size)
        numsamplexv = calcul_num_sample_multi_class(x_val_, this_size)

        print("Number of sample for multi class training:", numsamplex)
        print("Number of sample for cross validation:", numsamplexv)
        # train_data_generator_multi_class(x_train, y_train, numsample, n_dim, batch_size=256)
        history = model['net'].fit_generator(train_data_generator_multi_class(
            x_train_, y_train_, numsamplex, this_size,
            batch_size=batch_size),
            validation_data=cross_vld_data_generator_multi_class(x_val_, y_val_, numsamplexv, this_size, batch_size=batch_size),
            # validation_data = (x_val_, y_val_),
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=int(numsamplex / batch_size),
            validation_steps=int(numsamplexv / batch_size),
            verbose=settings['net_verbose'],
            callbacks=[ModelCheckpoint(model['weights'],
                                       # monitor=['val_all_loss'],
                                       save_best_only=True,
                                       save_weights_only=True),
                       EarlyStopping(  # monitor='val_loss',
                           # min_delta=0,
                           patience=settings['patience'])])

    model['history'] = history

    if settings['debug']:
        print("> DEBUG: loading weights after training")

    model['net'].load_weights(model['weights'])

    return model

def fit_thismodel(model, x_train, y_train, settings, initial_epoch=0):

    model['net'].load_weights(model['weights'])

    return model
