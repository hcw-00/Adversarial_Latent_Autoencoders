from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from dataset.mnist import load_mnist
import numpy as np
import pandas as pd
from collections import namedtuple

from module import *
from utils import *
import utils

import cv2

class vae(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        #self.image_size = args.fine_size
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.alpha = args.alpha

        self.f_encoder = f_encoder
        self.generator = generator
        self.e_encoder = e_encoder
        self.discriminator = discriminator
        self.mse = mse_criterion

        self._build_model(args)
        
        self.saver = tf.train.Saver(max_to_keep=100)
        
        if args.phase == 'train':
            (train_image_data, train_label_data), (test_image_data, test_label_data) = load_mnist(flatten = False, normalize = False)
        else:
            print('_')
        
    def _load_batch(self, dataset, idx):
        
        filename_list = dataset.iloc[:,0][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist()

        # input batch (2d binary image)
        input_batch = []
        for i in range(len(filename_list)):
            temp_img = cv2.imread('./dataset/rcwa_data_0608/64/'+filename_list[i], 0)
            temp_img = temp_img/128 - 1
            input_batch.append(list(temp_img))
        input_batch = np.expand_dims(input_batch, axis=3)

        # target batch (spectrum)
        target_batch = np.expand_dims(dataset.iloc[:,6:][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist(), 2) # [0.543, ... ] 226개
        target_batch = target_batch/180

        return input_batch, target_batch, filename_list


    def _build_model(self, args):

        self.real_input = tf.placeholder(tf.float32, [None,28,28,1], name='input')
        self.z_input = tf.placeholder(tf.float32, [None,512], name='z_input')

        #
        #z_input = tf.random_normal([512], 0, 1, dtype=tf.float32)
        w_F_fake = self.f_encoder(self.z_input, reuse=False, name='f_encoder')
        fake_ = self.generator(w_f_fake, eta, reuse=False, name='generator')
        w_E_fake = self.e_encoder(fake_, reuse=False, name='e_encoder')
        D_fake = self.discriminator(w_e_fake, reuse=False, name='discriminator')

        w_E_real = self.e_encoder(self.real_input, reuse=True, name='e_encoder')
        D_real = self.discriminator(w_e_real, reuse=True, name='discriminator')

        # losses
        self.ED_adv_loss = softplus(D_fake) + softplus(-D_real) # + "Gradient regularization term"
        self.FG_adv_loss = softplus(-D_fake)
        self.EG_loss = mse_criterion(w_F_fake, w_E_real)
        

        self.loss_summary = tf.summary.scalar("loss", self.total_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var]
        self.g_vars = [var for var in t_vars if 'generator' in var]
        self.e_vars = [var for var in t_vars if 'e_encoder' in var]
        self.f_vars = [var for var in t_vars if 'f_encoder' in var]
        print("trainable variables : ")
        print(t_vars)
        

    def train(self, args):
        
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 0.96, staircase=False)

        self.ED_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1) \
            .minimize(self.ED_adv_loss, var_list=[self.e_vars, self.d_vars], global_step = global_step)
        self.FG_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1) \
            .minimize(self.FG_adv_loss, var_list=[self.f_vars, self.g_vars], global_step = global_step)
        self.EG_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1) \
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
            
            batch_idxs = len(self.ds) // self.batch_size

            #ds_1 = self.ds.sample(frac=1)
            
            for idx in range(0, batch_idxs):

                input_batch, target_batch, _ = self._load_batch(self.ds, idx)

                z_input = np.random.normal(0,1,[self.batch_size,512])

                # Update E, D
                _ = self.sess.run([self.ED_optim], feed_dict={self.z_input : z_input, self.input : input_batch})
                # Update F, G
                _ = self.sess.run([self.FG_optim], feed_dict={self.z_input : z_input, self.input : input_batch})
                # Update E, G
                _ = self.sess.run([self.EG_optim], feed_dict={self.z_input : z_input, self.input : input_batch})

                self.writer.add_summary(summary_str, counter)

                counter += 1
                if idx%10==0:
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %4.4f loss_l: %4.4f loss_r: %4.4f lr: %4.7f kl: %4.7f m: %4.7f" % (
                        epoch, idx, batch_idxs, time.time() - start_time, loss,loss_l,loss_r, c_lr, np.mean(kl), np.mean(marginal))))

                if np.mod(counter, args.save_freq) == 20:
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
            temp_ckpt = 'dnn.model-80520'
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


        if not os.path.exists(os.path.join(args.test_dir,'input')):
            os.mkdir(os.path.join(args.test_dir,'input'))


        batch_idxs = len(self.ds) // self.batch_size

        #ds_1 = self.ds.sample(frac=1)
        ds_1 = self.ds
        #print(ds_1.iloc[:,4:][0:10].values.tolist())
        
        loss_list = []

        df_param_target_all = pd.DataFrame()
        df_param_pred_all = pd.DataFrame()

        for idx in range(0, batch_idxs):

            input_batch, target_batch, _ = self._load_batch(ds_1, idx)

            geo_pred, pred, loss = self.sess.run([self.geo_reconstructed_l, self.spectra_l_predicted, self.total_loss],
                                                feed_dict={self.geo_labeled: input_batch, self.spectrum_target: target_batch})


            loss_list.append(loss)

            counter += 1
            if idx%1==0:
                print(("Step: [%4d/%4d] time: %4.4f" % (
                    idx, batch_idxs, time.time() - start_time)))
                #df_param = pd.DataFrame(np.squeeze(input_batch), columns={'param1','param2','param3','param4','param5'}) 
                df_pred = pd.DataFrame(np.squeeze(pred))
                df_target = pd.DataFrame(np.squeeze(target_batch))
                #df_geo_pred =  np.squeeze(geo_pred)

                #df_param_pred = pd.concat([df_param, df_pred], axis=1, sort=False)
                #df_param_target = pd.concat([df_param, df_target], axis=1, sort=False)
                #df_param_param = pd.concat([df_param, df_geo_pred], axis=1, sort=False)
                
                df_param_target_all = pd.concat([df_param_target_all, df_target], axis=0, sort=False)
                df_param_pred_all = pd.concat([df_param_pred_all, df_pred], axis=0, sort=False)


            df_param_target_all.to_csv('./test/result_test_target.csv', index=False)
            df_param_pred_all.to_csv('./test/result_test_prediction.csv', index=False)

            #print(np.shape(geo_pred))
            geo_pred = np.squeeze(geo_pred)
            #print(geo_pred)
            #cv2.imwrite('./test/reconstructed/test'+str(idx)+'.bmp',(geo_pred+1)*128)
            
        print("loss")
        print(np.mean(loss_list))
        print("total time")
        print(time.time() - start_time)


    def test_reconstruction(self, args):

        self.batch_size = 1

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0


        if not os.path.exists(os.path.join(args.test_dir,'input')):
            os.mkdir(os.path.join(args.test_dir,'input'))


        batch_idxs = len(self.ds) // self.batch_size

        #ds_1 = self.ds.sample(frac=1)
        ds_1 = self.ds
        #print(ds_1.iloc[:,4:][0:10].values.tolist())
        
        loss_list = []

        for idx in range(0, batch_idxs):

            input_batch, target_batch, filename_list = self._load_batch(ds_1, idx)

            for j in range(5):
                latent_vector = list(np.random.normal(0,3,5))
                #for k in range(5):
                #    latent_vector[k] = j*0.5 - 2.5
                print(latent_vector)
                latent_vector = np.expand_dims(latent_vector, 0)
                geo_recon = self.sess.run([self.geo_reconstructed], 
                                            feed_dict={self.latent_vector: latent_vector, self.spectrum_target: target_batch})


            