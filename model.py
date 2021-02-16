import os
import sys

import math
import time
import argparse
import numpy as np
import tensorflow as tf


fc = tf.keras.layers.Dense
w_init = tf.random_normal_initializer(stddev=0.02)
b_init = tf.constant_initializer(0.0)


class Seq2Seq:
    def __init__(self, cell, embeddings, role):
        self.cell = cell
        self.embeddings = embeddings
        self.role = role
        if role not in ['encoder', 'decoder']:
            raise ValueError("The role must be 'encoder' or 'decoder'.")

    def __call__(self, tokens, seq_lengths, initial_state=None):
        token_embeds = tf.nn.embedding_lookup(self.embeddings, tokens)
        outputs, final_state = tf.nn.dynamic_rnn(
            self.cell, token_embeds,
            initial_state=initial_state,
            sequence_length=seq_lengths,
            dtype=tf.float32)
        if self.role == 'encoder':
            return final_state
        elif self.role == 'decoder':
            return outputs


class LatentGaussianMixture:
    def __init__(self, args):
        self.args = args
        with tf.variable_scope("priors"):
            if args.pretrain_dir:
                mu_c_path = '{}/{}_{}_{}_{}/init_mu_c.npz'.format(args.pretrain_dir, args.model, 
                        args.token_dim, args.rnn_dim, args.cluster_num)
                mu_c = np.load(mu_c_path)['arr_0']
                self.mu_c = tf.get_variable("mu_c", initializer=tf.constant(mu_c))
            else:
                self.mu_c = tf.get_variable("mu_c", [args.cluster_num, args.rnn_dim],
                        initializer=tf.random_uniform_initializer(0.0, 1.0))

            self.log_sigma_sq_c = tf.get_variable("sigma_sq_c", [args.cluster_num, args.rnn_dim],
                    initializer=tf.constant_initializer(0.0), trainable=False)

            # log_pi_prior = tf.get_variable("log_pi_prior", args.cluster_num,
            #                                initializer=tf.constant_initializer(0.0), trainable=False)
            # self.pi_prior = tf.nn.softmax(log_pi_prior)

        with tf.variable_scope("compute_posteriors"):
            self.fc_mu_z = fc(args.rnn_dim, activation=None, use_bias=True,
                    kernel_initializer=w_init, bias_initializer=b_init)

            self.fc_sigma_z = fc(args.rnn_dim, activation=None, use_bias=True,
                    kernel_initializer=w_init, bias_initializer=b_init)

    def post_sample(self, embeded_state, return_loss=False):
        args = self.args

        mu_z = self.fc_mu_z(embeded_state)
        log_sigma_sq_z = self.fc_sigma_z(embeded_state)

        eps_z = tf.random_normal(shape=tf.shape(log_sigma_sq_z), mean=0.0, stddev=1.0, dtype=tf.float32)
        z = mu_z + tf.sqrt(tf.exp(log_sigma_sq_z)) * eps_z

        stack_z = tf.stack([z] * args.cluster_num, axis=1)
        stack_mu_c = tf.stack([self.mu_c] * args.batch_size, axis=0)
        stack_mu_z = tf.stack([mu_z] * args.cluster_num, axis=1)
        stack_log_sigma_sq_c = tf.stack([self.log_sigma_sq_c] * args.batch_size, axis=0)
        stack_log_sigma_sq_z = tf.stack([log_sigma_sq_z] * args.cluster_num, axis=1)

        pi_post_logits = - tf.reduce_sum(tf.square(stack_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c), axis=-1)
        pi_post = tf.nn.softmax(pi_post_logits) + 1e-10

        if not return_loss:
            return z
        else:
            batch_gaussian_loss = 0.5 * tf.reduce_sum(
                    pi_post * tf.reduce_mean(stack_log_sigma_sq_c
                        + tf.exp(stack_log_sigma_sq_z) / tf.exp(stack_log_sigma_sq_c)
                        + tf.square(stack_mu_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c), axis=-1)
                    , axis=-1) - 0.5 * tf.reduce_mean(1 + log_sigma_sq_z, axis=-1)

            batch_uniform_loss = tf.reduce_mean(tf.reduce_mean(pi_post, axis=0) * tf.log(tf.reduce_mean(pi_post, axis=0)))
            return z, [batch_gaussian_loss, batch_uniform_loss]

    def prior_sample(self):
        pass


class Model:
    def __init__(self, args):
        self.args = args
        x_size = args.map_size[0] * args.map_size[1]
        self.out_size = out_size = x_size

        with tf.variable_scope("embeddings"):
            embeddings = tf.Variable(
                    tf.random_uniform([x_size, args.token_dim], -1.0, 1.0),
                    dtype=tf.float32)

        with tf.variable_scope("encoder"):
            encoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_dim, name='enc_cell')
            self.encoder = Seq2Seq(encoder_cell, embeddings, 'encoder')

        with tf.variable_scope("latent_space"):
            self.latent_space = LatentGaussianMixture(args)

        with tf.variable_scope("decoder"):
            decoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_dim, name='dec_cell')
            self.decoder = Seq2Seq(decoder_cell, embeddings, 'decoder')

        with tf.variable_scope("output"):
            self.out_w = tf.get_variable("out_w", [out_size, args.rnn_dim], tf.float32, initializer=w_init)
            self.out_b = tf.get_variable("out_b", [out_size], tf.float32, initializer=b_init)

    def __call__(self, inputs):
        args = self.args

        tokens, masks, seq_lengths = inputs
        encoder_final_state = self.encoder(tokens, seq_lengths)

        z, latent_losses = self.latent_space.post_sample(encoder_final_state, return_loss=True)

        batch_zeros = tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32)
        targets = tf.concat([tokens, batch_zeros], axis=1)
        tokens = tf.concat([batch_zeros, tokens], axis=1)
        masks = tf.concat([masks, batch_zeros], axis=1)

        outputs = self.decoder(tokens, seq_lengths, initial_state=z)

        if args.mode == 'train' or args.mode == 'pretrain':
            res = self.loss(outputs, targets, masks, latent_losses)
            res += [z]
        elif args.mode == 'eval':
            res = self.anomaly_score(outputs, targets, masks)
        return res

    def anomaly_score(self, outputs, targets, masks):
        masks = tf.cast(masks, tf.float32)
        target_out_w = tf.nn.embedding_lookup(self.out_w, targets)
        target_out_b = tf.nn.embedding_lookup(self.out_b, targets)
        score = tf.reduce_sum(
                masks * tf.exp(tf.log_sigmoid(
                    tf.reduce_sum(outputs * target_out_w, axis=-1) + target_out_b
                    )), axis=-1, name="anomaly_score") / tf.reduce_sum(masks, axis=-1)
        return score

    def loss(self, outputs, targets, masks, latent_losses):
        args = self.args
        batch_gaussian_loss, batch_uniform_loss = latent_losses
        batch_rec_loss = tf.reduce_mean(
                tf.cast(masks, tf.float32) * tf.reshape(
                    tf.nn.sampled_softmax_loss(
                        weights=self.out_w,
                        biases=self.out_b,
                        labels=tf.reshape(targets, [-1, 1]),
                        inputs=tf.reshape(outputs, [-1, args.rnn_dim]),
                        num_sampled=args.num_negs,
                        num_classes=self.out_size
                    ), [args.batch_size, -1]
                ), axis=-1)

        rec_loss = tf.reduce_mean(batch_rec_loss)
        gaussian_loss = tf.reduce_mean(batch_gaussian_loss)
        uniform_loss = tf.reduce_mean(batch_uniform_loss)

        if args.cluster_num == 1:
            loss = rec_loss + gaussian_loss
        else:
            loss = 1.0 * rec_loss + 1.0 / args.rnn_dim * gaussian_loss + 0.1 * uniform_loss
            # loss = 1.0 * rec_loss + 1.0 * gaussian_loss + 0.1 * uniform_loss

        pretrain_loss = rec_loss
        return [loss, pretrain_loss]

