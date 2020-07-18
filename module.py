from __future__ import division
import tensorflow as tf
from ops import *
from utils import *



def f_encoder(inputs, reuse=False, name="f_encoder"):

    with tf.variable_scope("f_encoder"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 50, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net

def generator(inputs, eta, reuse=False, name="generator"):

    with tf.variable_scope("generator"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 784, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net

def e_encoder(inputs, reuse=False, name="e_encoder"):

    with tf.variable_scope("e_encoder"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 50, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net

def discriminator(inputs, reuse=False, name="discriminator"):

    with tf.variable_scope("discriminator"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.initializers.he_normal())
        net = slim.fully_connected(net, 1, activation_fn=None, weights_initializer=tf.initializers.he_normal())

        return net



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
