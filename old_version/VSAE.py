import os
import sys
sys.path.append("../")

import time
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator

from sklearn.metrics import precision_recall_curve, auc


def auc_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(1-y_true, 1-y_score)
    return auc(recall, precision)


def filling_batch(batch_data):
    new_batch_data = []
    last_batch_size = len(batch_data[0])
    for b in batch_data:
        new_batch_data.append(
            np.concatenate([b, [np.zeros_like(b[0]).tolist()
                                for _ in range(args.batch_size - last_batch_size)]], axis=0))
    return new_batch_data


def compute_likelihood(sess, model, sampler, purpose):
    all_likelihood = []
    for batch_data in sampler.iterate_all_data(args.batch_size,
                                               partial_ratio=args.partial_ratio,
                                               purpose=purpose):
        if len(batch_data[0]) < args.batch_size:
            last_batch_size = len(batch_data[0])
            batch_data = filling_batch(batch_data)
            feed = dict(zip(model.input_form, batch_data))
            batch_likelihood = sess.run(model.batch_likelihood, feed)[:last_batch_size]
        else:
            feed = dict(zip(model.input_form, batch_data))
            batch_likelihood = sess.run(model.batch_likelihood, feed)
        all_likelihood.append(batch_likelihood)
    return np.concatenate(all_likelihood)


def compute_loss(sess, model, sampler, purpose):
    all_loss = []
    for batch_data in sampler.iterate_all_data(args.batch_size,
                                               partial_ratio=args.partial_ratio,
                                               purpose=purpose):
        if len(batch_data[0]) < args.batch_size:
            batch_data = filling_batch(batch_data)
            feed = dict(zip(model.input_form, batch_data))
            loss = sess.run(model.loss, feed)
        else:
            feed = dict(zip(model.input_form, batch_data))
            loss = sess.run(model.loss, feed)
        all_loss.append(loss)
    return np.mean(all_loss)


class Model:
    def __init__(self, args):
        inputs = tf.placeholder(shape=(args.batch_size, None), dtype=tf.int32, name='inputs')
        mask = tf.placeholder(shape=(args.batch_size, None), dtype=tf.float32, name='inputs_mask')
        seq_length = tf.placeholder(shape=args.batch_size, dtype=tf.float32, name='seq_length')

        self.input_form = [inputs, mask, seq_length]

        encoder_inputs = inputs
        decoder_inputs = tf.concat([tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32), inputs], axis=1)
        decoder_targets = tf.concat([inputs, tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32)], axis=1)
        decoder_mask = tf.concat([mask, tf.zeros(shape=(args.batch_size, 1), dtype=tf.float32)], axis=1)

        x_size = out_size = args.map_size[0] * args.map_size[1]
        embeddings = tf.Variable(tf.random_uniform([x_size, args.x_latent_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

        with tf.variable_scope("encoder"):
            encoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size)
            _, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embedded,
                sequence_length=seq_length,
                dtype=tf.float32,
            )

        mu_w = tf.get_variable("mu_w", [args.rnn_size, args.rnn_size], tf.float32,
                               tf.random_normal_initializer(stddev=0.02))
        mu_b = tf.get_variable("mu_b", [args.rnn_size], tf.float32,
                               initializer=tf.constant_initializer(0.0))
        sigma_w = tf.get_variable("sigma_w", [args.rnn_size, args.rnn_size], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))
        sigma_b = tf.get_variable("sigma_b", [args.rnn_size], tf.float32,
                                  initializer=tf.constant_initializer(0.0))

        mu = tf.matmul(encoder_final_state, mu_w) + mu_b
        log_sigma_sq = tf.matmul(encoder_final_state, sigma_w) + sigma_b
        eps = tf.random_normal(shape=tf.shape(log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)

        if args.eval:
            z = tf.zeros(shape=(args.batch_size, args.rnn_size), dtype=tf.float32)
        else:
            z = mu + tf.sqrt(tf.exp(log_sigma_sq)) * eps

        self.batch_post_embedded = z

        with tf.variable_scope("decoder"):
            decoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size)
            decoder_init_state = z
            decoder_outputs, _ = tf.nn.dynamic_rnn(
                decoder_cell, decoder_inputs_embedded,
                initial_state=decoder_init_state,
                sequence_length=seq_length,
                dtype=tf.float32,
            )

        out_w = tf.get_variable("out_w", [out_size, args.rnn_size], tf.float32,
                                tf.random_normal_initializer(stddev=0.02))
        out_b = tf.get_variable("out_b", [out_size], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        batch_rec_loss = tf.reduce_mean(
            decoder_mask * tf.reshape(
                tf.nn.sampled_softmax_loss(
                    weights=out_w,
                    biases=out_b,
                    labels=tf.reshape(decoder_targets, [-1, 1]),
                    inputs=tf.reshape(decoder_outputs, [-1, args.rnn_size]),
                    num_sampled=args.neg_size,
                    num_classes=out_size
                ), [args.batch_size, -1]
            ), axis=-1
        )
        batch_latent_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), axis=1)

        self.rec_loss = rec_loss = tf.reduce_mean(batch_rec_loss)
        self.latent_loss = latent_loss = tf.reduce_mean(batch_latent_loss)

        self.loss = loss = tf.reduce_mean([rec_loss, latent_loss])
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

        target_out_w = tf.nn.embedding_lookup(out_w, decoder_targets)
        target_out_b = tf.nn.embedding_lookup(out_b, decoder_targets)

        self.batch_likelihood = tf.reduce_mean(
            decoder_mask * tf.log_sigmoid(
                tf.reduce_sum(decoder_outputs * target_out_w, -1) + target_out_b
            ), axis=-1, name="batch_likelihood")

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
        self.save, self.restore = saver.save, saver.restore


def train():
    model = Model(args)
    sampler = DataGenerator(args)

    all_val_loss = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        start = time.time()
        for epoch in range(args.num_epochs):
            all_loss = []
            # sampler.spatial_augmentation()
            for batch_idx in range(int(sampler.total_traj_num / args.batch_size)):
                batch_data = sampler.next_batch(args.batch_size)
                feed = dict(zip(model.input_form, batch_data))

                rec_loss, latent_loss, _ = sess.run(
                    [model.rec_loss, model.latent_loss, model.train_op], feed)
                all_loss.append([rec_loss, latent_loss])

            val_loss = compute_loss(sess, model, sampler, "val")
            if len(all_val_loss) > 0 and val_loss >= all_val_loss[-1]:
                print("Early termination with val loss: {}:".format(val_loss))
                break

            all_val_loss.append(val_loss)

            end = time.time()
            print("epoch: {}\tval loss: {}\telapsed time: {}".format(epoch, val_loss, end - start))
            print("loss: {}".format(np.mean(all_loss, axis=0)))
            start = time.time()

            save_model_name = "./models/{}_{}_{}/{}_{}".format(
                args.model_type, args.x_latent_size, args.rnn_size, args.model_type, epoch)
            model.save(sess, save_model_name)


def evaluate():
    model = Model(args)
    sampler = DataGenerator(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, args.model_id)
        model.restore(sess, model_name)

        st = time.time()
        all_likelihood = compute_likelihood(sess, model, sampler, "train")
        elapsed = time.time() - st

        all_prob = np.exp(all_likelihood)

        y_true = np.ones_like(all_prob)
        for idx in sampler.outliers:
            if idx < y_true.shape[0]:
                y_true[idx] = 0

        sd_auc = {}
        sd_index = sampler.sd_index
        for sd, tids in sd_index.items():
            sd_y_true = y_true[tids]
            sd_prob = all_prob[tids]
            if sd_y_true.sum() < len(sd_y_true):
                sd_auc[sd] = auc_score(y_true=sd_y_true, y_score=sd_prob)
        print("Average AUC:", np.mean(list(sd_auc.values())), "Elapsed time:", elapsed)

        sorted_sd_index = sorted(list(sd_auc.keys()), key=lambda k: len(sd_index[k]))
        sorted_sd_auc = [sd_auc[sd] for sd in sorted_sd_index]

        bin_num = 5
        step_size = int(len(sorted_sd_auc) / bin_num)
        for i in range(bin_num):
            print(np.mean(sorted_sd_auc[i*step_size:(i+1)*step_size]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="../data/processed_beijing{}.csv",
                        help='data file')
    parser.add_argument('--map_size', type=tuple, default=(130, 130),
                        help='size of map')
    # parser.add_argument('--data_filename', type=str, default="../data/processed_porto{}.csv",
    #                     help='data file')
    # parser.add_argument('--map_size', type=tuple, default=(51, 158),
    #                     help='size of map')
    parser.add_argument('--model_type', type=str, default="seq2seq",
                        help='choose a model')
    parser.add_argument('--x_latent_size', type=int, default=32,
                        help='size of input embedding')
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--neg_size', type=int, default=64,
                        help='size of negative sampling')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')

    parser.add_argument('--model_id', type=str, default="",
                        help='model id')
    parser.add_argument('--partial_ratio', type=float, default=1.0,
                        help='partial trajectory evaluation')
    parser.add_argument('--eval', type=bool, default=False,
                        help='partial trajectory evaluation')

    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.eval:
        evaluate()
    else:
        train()

