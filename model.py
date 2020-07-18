from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from dataset.mnist import load_mnist
import numpy as np
import pandas as pd
#from collections import namedtuple
from sklearn.utils import shuffle
from module import *
from utils import *
import utils

import cv2

#TODO : add noise input eta

class vae(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        #self.image_size = args.fine_size
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.alpha = args.alpha
        self.gamma = 10
        self.f_encoder = f_encoder
        self.generator = generator
        self.e_encoder = e_encoder
        self.discriminator = discriminator
        self.mse = mse_criterion

        self._build_model(args)
        
        self.saver = tf.train.Saver(max_to_keep=100)
        
        
        (self.train_image, self.train_label), (self.test_image, self.test_label) = load_mnist(flatten = True, normalize = True)

    def _load_batch(self, idx, images):

        input_batch = []
        #target_batch = []
        for i in range(self.batch_size):
            input_batch.append(np.squeeze(images[i+self.batch_size*idx]))
            #target_batch.append(self.train_label[i+self.batch_size*idx])
        #input_batch = np.expand_dims(input_batch, axis=3)

        return input_batch#, target_batch


    def _build_model(self, args):

        self.real_input = tf.placeholder(tf.float32, [None,784], name='input')
        self.z_input = tf.placeholder(tf.float32, [None,128], name='z_input')

        #
        eta = tf.random.normal([1])
        #z_input = tf.random_normal([512], 0, 1, dtype=tf.float32)
        w_F_fake = self.f_encoder(self.z_input, reuse=False, name='f_encoder')
        self.fake_ = self.generator(w_F_fake, eta, reuse=False, name='generator')
        w_E_fake = self.e_encoder(self.fake_, reuse=False, name='e_encoder')
        D_fake = self.discriminator(w_E_fake, reuse=False, name='discriminator')

        w_E_real = self.e_encoder(self.real_input, reuse=True, name='e_encoder')
        D_real = self.discriminator(w_E_real, reuse=True, name='discriminator')        

        # inference network
        w_test = self.e_encoder(self.real_input, reuse=True, name='e_encoder')
        self.recon_image = self.generator(w_test, eta, reuse=True, name='generator')

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.e_vars = [var for var in t_vars if 'e_encoder' in var.name]
        self.f_vars = [var for var in t_vars if 'f_encoder' in var.name]
        print("trainable variables : ")
        print(t_vars)

        # losses
        # dc_dw, dc_db = tf.gradients(cost, [W, b])
        self.D_real_loss = tf.reduce_mean(softplus(-D_real))
        self.D_fake_loss = tf.reduce_mean(softplus(D_fake))
        self.D_grad_loss = self.grad_reg(self.D_real_loss, self.e_vars+self.d_vars) # gradient regularization term
        self.ED_adv_loss = self.D_fake_loss + self.D_real_loss #+ self.D_grad_loss
        self.FG_adv_loss = tf.reduce_mean(softplus(-D_fake))
        self.EG_loss = mse_criterion(w_F_fake, w_E_fake)

        self.loss_summary = tf.summary.scalar("loss", self.ED_adv_loss)
        
    def grad_reg(self, t, vars):
        grad = tf.gradients(t, vars)
        gradreg = self.gamma/2 * tf.reduce_mean([tf.reduce_mean(tf.square(g)) for g in grad])
        return gradreg

    def train(self, args):
        
        #self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.lr = args.lr
        
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 0.96, staircase=False)
        learning_rate = self.lr

        self.ED_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.ED_adv_loss, var_list=[self.e_vars, self.d_vars], global_step = global_step)
        self.FG_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.FG_adv_loss, var_list=[self.f_vars, self.g_vars], global_step = global_step)
        self.EG_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.EG_loss, var_list=[self.e_vars, self.g_vars], global_step = global_step)

        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir): 
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            
            batch_idxs = len(self.train_label) // self.batch_size

            self.train_image, self.train_label = shuffle(self.train_image, self.train_label)
            
            for idx in range(0, batch_idxs):

                # Update E, D
                input_batch = self._load_batch(idx, self.train_image)
                z_input = np.random.normal(0,1,[self.batch_size,128])
                _, ed_loss = self.sess.run([self.ED_optim, self.ED_adv_loss], feed_dict={self.z_input : z_input, self.real_input : input_batch})
                # Update F, G
                z_input = np.random.normal(0,1,[self.batch_size,128])
                _, fg_loss, fake_img = self.sess.run([self.FG_optim, self.FG_adv_loss, self.fake_], feed_dict={self.z_input : z_input, self.real_input : input_batch})
                # Update E, G
                z_input = np.random.normal(0,1,[self.batch_size,128])
                _, eg_loss = self.sess.run([self.EG_optim, self.EG_loss], feed_dict={self.z_input : z_input, self.real_input : input_batch})

                #self.writer.add_summary(summary_str, counter)

                counter += 1
                if idx%20==0:
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f ed adv loss: %4.4f fg adv loss: %4.4f eg loss: %4.4f" % (
                        epoch, idx, batch_idxs, time.time() - start_time, ed_loss, fg_loss, eg_loss)))

                if idx == batch_idxs-1:
                    #self.save(args.checkpoint_dir, counter)
                    temp_fake = np.reshape(fake_img[0]*255, (28,28))
                    #temp_recon = np.reshape(input_batch[j]*255, (28,28))
                    cv2.imwrite('./sample/fake_'+str(epoch)+'.bmp', temp_fake)
                    #cv2.imwrite('./test/recon_'+str(j)+'.bmp', temp_image)
                if epoch == args.epoch-1 and idx == batch_idxs-1:
                    self.save(args.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths    #hcw
            print(ckpt_paths)
            #ckpt_name = os.path.basename(ckpt_paths[-1])    #hcw # default [-1]
            temp_ckpt = 'dnn.model-23401'
            ckpt_name = os.path.basename(temp_ckpt)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        
        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0

        for epoch in range(1):
            
            batch_idxs = len(self.test_label) // self.batch_size

            #self.train_image, self.train_label = shuffle(self.train_image, self.train_label)
            
            for idx in range(1):

                input_batch = self._load_batch(idx, self.test_image)

                z_input = np.random.normal(0,1,[self.batch_size,128])

                recon_img = self.sess.run([self.recon_image], feed_dict={self.z_input : z_input, self.real_input : input_batch})

                for j in range(len(recon_img[0])):
                    temp_image = np.reshape(recon_img[0][j]*255, (28,28))
                    temp_real = np.reshape(input_batch[j]*255, (28,28))
                    cv2.imwrite('./test/'+str(j)+'_real.bmp', temp_real)
                    cv2.imwrite('./test/'+str(j)+'_recon.bmp', temp_image)
                counter += 1


            