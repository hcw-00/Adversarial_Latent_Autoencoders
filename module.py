from __future__ import division
import tensorflow as tf
from ops import *
from utils import *



def f_encoder(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        inputs = slim.flatten(inputs)
        net_a = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)

        net_a = slim.fully_connected(net_a, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(net_b, 512, activation_fn=tf.nn.tanh)

        spectrum_a = slim.fully_connected(net_a, 101, activation_fn=tf.nn.tanh)
        spectrum_b = slim.fully_connected(net_b, 101, activation_fn=tf.nn.tanh)

        spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
        spectra = tf.expand_dims(spectra, axis=2)
        return spectra

def generator(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        inputs = slim.flatten(inputs)
        net_a = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)

        net_a = slim.fully_connected(net_a, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(net_b, 512, activation_fn=tf.nn.tanh)

        spectrum_a = slim.fully_connected(net_a, 101, activation_fn=tf.nn.tanh)
        spectrum_b = slim.fully_connected(net_b, 101, activation_fn=tf.nn.tanh)

        spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
        spectra = tf.expand_dims(spectra, axis=2)
        return spectra

def e_encoder(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        inputs = slim.flatten(inputs)
        net_a = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)

        net_a = slim.fully_connected(net_a, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(net_b, 512, activation_fn=tf.nn.tanh)

        spectrum_a = slim.fully_connected(net_a, 101, activation_fn=tf.nn.tanh)
        spectrum_b = slim.fully_connected(net_b, 101, activation_fn=tf.nn.tanh)

        spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
        spectra = tf.expand_dims(spectra, axis=2)
        return spectra

def discriminator(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        inputs = slim.flatten(inputs)
        net_a = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)

        net_a = slim.fully_connected(net_a, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(net_b, 512, activation_fn=tf.nn.tanh)

        spectrum_a = slim.fully_connected(net_a, 101, activation_fn=tf.nn.tanh)
        spectrum_b = slim.fully_connected(net_b, 101, activation_fn=tf.nn.tanh)

        spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
        spectra = tf.expand_dims(spectra, axis=2)
        return spectra



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
