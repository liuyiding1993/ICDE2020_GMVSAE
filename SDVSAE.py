import os
import sys
sys.path.append("../")

import time
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator

from sklearn.metrics import precision_recall_curve, auc
from sklearn.cluster import KMeans


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


def compute_output(output_type, sess, model, sampler, purpose, callback):
    all_output = []
    for batch_data, batch_sd in sampler.iterate_all_data(args.batch_size,
                                                         partial_ratio=args.partial_ratio,
                                                         purpose=purpose):
        batch_s, batch_d = batch_sd
        if len(batch_data[0]) < args.batch_size:
            last_batch_size = len(batch_data[0])
            batch_data = filling_batch(batch_data)
            feed = dict(zip(model.input_form, batch_data))
            feed[model.s_inputs] = batch_s + [0] * (args.batch_size - last_batch_size)
            feed[model.d_inputs] = batch_d + [0] * (args.batch_size - last_batch_size)
            batch_output = sess.run(output_type, feed)
            if type(batch_output) is list and len(batch_output) == last_batch_size:
                batch_output = batch_output[: last_batch_size]
        else:
            feed = dict(zip(model.input_form, batch_data))
            feed[model.s_inputs] = batch_s
            feed[model.d_inputs] = batch_d
            batch_output = sess.run(output_type, feed)
        all_output.append(batch_output)
    return callback(all_output)


class Model:
    def __init__(self, args):
        self.args = args
        dense = tf.layers.dense

        inputs = tf.placeholder(shape=(args.batch_size, None), dtype=tf.int32, name='inputs')
        mask = tf.placeholder(shape=(args.batch_size, None), dtype=tf.float32, name='inputs_mask')
        seq_length = tf.placeholder(shape=args.batch_size, dtype=tf.float32, name='seq_length')

        self.s_inputs = s_inputs = tf.placeholder(shape=args.batch_size, dtype=tf.int32, name='s_inputs')
        self.d_inputs = d_inputs = tf.placeholder(shape=args.batch_size, dtype=tf.int32, name='d_inputs')

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

        with tf.variable_scope("clusters"):
            mu_c = tf.get_variable("mu_c", [args.mem_num, args.rnn_size],
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            log_sigma_sq_c = tf.get_variable("sigma_sq_c", [args.mem_num, args.rnn_size],
                                             initializer=tf.constant_initializer(0.0), trainable=False)
            log_pi_prior = tf.get_variable("log_pi_prior", args.mem_num,
                                           initializer=tf.constant_initializer(0.0), trainable=False)
            pi_prior = tf.nn.softmax(log_pi_prior)

            init_mu_c = tf.placeholder(shape=(args.mem_num, args.rnn_size), dtype=tf.float32, name='init_mu_c')
            init_sigma_c = tf.placeholder(shape=(args.mem_num, args.rnn_size), dtype=tf.float32, name='init_sigma_c')
            init_pi = tf.placeholder(shape=args.mem_num, dtype=tf.float32, name='init_pi')
            self.cluster_init = [init_mu_c, init_sigma_c, init_pi]

            self.init_mu_c_op = tf.assign(mu_c, init_mu_c)
            self.init_sigma_c_op = tf.assign(log_sigma_sq_c, init_sigma_c)
            self.init_pi_op = tf.assign(log_pi_prior, init_pi)

            self.mu_c = mu_c
            self.sigma_c = log_sigma_sq_c
            self.pi = pi_prior

            stack_mu_c = tf.stack([mu_c] * args.batch_size, axis=0)
            stack_log_sigma_sq_c = tf.stack([log_sigma_sq_c] * args.batch_size, axis=0)

        with tf.variable_scope("latent"):
            mu_z = dense(encoder_final_state, args.rnn_size, activation=None)
            log_sigma_sq_z = dense(encoder_final_state, args.rnn_size, activation=None)

            eps_z = tf.random_normal(shape=tf.shape(log_sigma_sq_z), mean=0, stddev=1, dtype=tf.float32)
            z = mu_z + tf.sqrt(tf.exp(log_sigma_sq_z)) * eps_z

            stack_mu_z = tf.stack([mu_z] * args.mem_num, axis=1)
            stack_log_sigma_sq_z = tf.stack([log_sigma_sq_z] * args.mem_num, axis=1)
            stack_z = tf.stack([z] * args.mem_num, axis=1)
            self.batch_post_embedded = z

        with tf.variable_scope("sd_attention"):
            s_embeddings = tf.Variable(tf.random_uniform([x_size, args.rnn_size], -1.0, 1.0), dtype=tf.float32)
            d_embeddings = tf.Variable(tf.random_uniform([x_size, args.rnn_size], -1.0, 1.0), dtype=tf.float32)
            s = tf.nn.embedding_lookup(s_embeddings, s_inputs)
            d = tf.nn.embedding_lookup(d_embeddings, d_inputs)
            sd = tf.concat([s, d], axis=1)
            hsd1 = dense(sd, args.rnn_size, activation=tf.nn.relu)
            sd_logits = dense(hsd1, args.mem_num, activation=tf.nn.relu)
            sd_att = tf.nn.softmax(sd_logits)

        with tf.variable_scope("attention"):
            att_logits = - tf.reduce_sum(tf.square(stack_z - stack_mu_c)
                                         / tf.exp(stack_log_sigma_sq_c), axis=-1)
            att = tf.nn.softmax(att_logits) + 1e-10
            self.batch_att = att

        def generation(h):
            with tf.variable_scope("generation", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("decoder"):
                    decoder_init_state = h
                    decoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size)
                    decoder_outputs, _ = tf.nn.dynamic_rnn(
                        decoder_cell, decoder_inputs_embedded,
                        initial_state=decoder_init_state,
                        sequence_length=seq_length,
                        dtype=tf.float32,
                    )
                with tf.variable_scope("outputs"):
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
                    target_out_w = tf.nn.embedding_lookup(out_w, decoder_targets)
                    target_out_b = tf.nn.embedding_lookup(out_b, decoder_targets)
                    batch_likelihood = tf.reduce_mean(
                        decoder_mask * tf.log_sigmoid(
                            tf.reduce_sum(decoder_outputs * target_out_w, -1) + target_out_b
                        ), axis=-1, name="batch_likelihood")

                    batch_latent_loss = 0.5 * tf.reduce_sum(
                        att * tf.reduce_mean(stack_log_sigma_sq_c
                                             + tf.exp(stack_log_sigma_sq_z) / tf.exp(stack_log_sigma_sq_c)
                                             + tf.square(stack_mu_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c),
                                             axis=-1),
                        axis=-1) - 0.5 * tf.reduce_mean(1 + log_sigma_sq_z, axis=-1)
                    batch_cate_loss = tf.reduce_mean(tf.reduce_mean(att, axis=0) * tf.log(tf.reduce_mean(att, axis=0)))
                return batch_rec_loss, batch_latent_loss, batch_cate_loss, batch_likelihood

        if args.eval:
            sd_z = tf.matmul(
                tf.one_hot(tf.argmax(sd_att, axis=-1), depth=args.mem_num, axis=-1), mu_c)
            # sd_z = tf.matmul(
            #     tf.one_hot(tf.argmax(att-1e-10, axis=-1), depth=args.mem_num, axis=-1), mu_c)
            results = generation(sd_z)
            self.batch_likelihood = results[-1]
        else:
            results = generation(z)
            self.batch_likelihood = results[-1]
            self.rec_loss = rec_loss = tf.reduce_mean(results[0])
            self.latent_loss = latent_loss = tf.reduce_mean(results[1])
            self.cate_loss = cate_loss = results[2]

            self.sd_loss = sd_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=att, logits=sd_logits))

            self.loss = loss = rec_loss + latent_loss + 0.1* cate_loss
            self.pretrain_loss = pretrain_loss = rec_loss

            all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            sd_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sd_attention')
            cluster_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='clusters')
            vae_vars = list(set(all_vars) - set(sd_vars) - set(cluster_vars))

            self.pretrain_op = tf.train.AdamOptimizer(args.learning_rate).minimize(pretrain_loss, var_list=vae_vars)
            self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, var_list=vae_vars)
            self.sd_train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(sd_loss, var_list=sd_vars)

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=100)
        self.save, self.restore = saver.save, saver.restore


def pretrain():
    model = Model(args)
    sampler = DataGenerator(args)

    all_val_loss = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        start = time.time()

        for epoch in range(args.num_epochs):
            for batch_idx in range(int(sampler.total_traj_num / args.batch_size)):
                batch_data, batch_sd = sampler.next_batch(args.batch_size)
                feed = dict(zip(model.input_form, batch_data))
                sess.run(model.pretrain_op, feed)

            val_loss = compute_output(model.pretrain_loss, sess, model, sampler,
                                      purpose="val", callback=np.mean)
            if len(all_val_loss) > 0 and val_loss >= all_val_loss[-1]:
                print("Early termination with val loss: {}:".format(val_loss))
                break
            all_val_loss.append(val_loss)

            end = time.time()
            print("pretrain epoch: {}\tval loss: {}\telapsed time: {}".format(
                epoch, val_loss, end - start))
            start = time.time()

            save_model_name = "./models/{}_{}_{}/{}_{}".format(
                args.model_type, args.x_latent_size, args.rnn_size, args.model_type, "pretrain")
            model.save(sess, save_model_name)

        # model_name = "./models/{}_{}_{}/{}_{}".format(
        #     args.model_type, args.x_latent_size, args.rnn_size, args.model_type, 'pretrain')
        # model.restore(sess, model_name)

        # K-means init
        sample_num = 10000
        x_embedded = []
        for batch_idx in range(int(sample_num / args.batch_size)):
            batch_data, batch_sd = sampler.next_batch(args.batch_size)
            feed = dict(zip(model.input_form, batch_data))
            x_embedded.append(sess.run(model.batch_post_embedded, feed))
        x_embedded = np.concatenate(x_embedded, axis=0)
        kmeans = KMeans(n_clusters=args.mem_num)
        kmeans.fit(x_embedded)
        init_mu_c = kmeans.cluster_centers_
        init_sigma_c = np.zeros_like(init_mu_c)
        init_pi = np.zeros(shape=args.mem_num)

        init_feed = dict(zip(model.cluster_init, [init_mu_c, init_sigma_c, init_pi]))
        sess.run([model.init_mu_c_op, model.init_sigma_c_op, model.init_pi_op], init_feed)

        save_model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, "init")
        model.save(sess, save_model_name)

        print("Init model saved.")


def train():
    model = Model(args)
    sampler = DataGenerator(args)

    all_val_loss = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        start = time.time()

        model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, 'init')
        model.restore(sess, model_name)

        for epoch in range(args.num_epochs):
            all_loss = []
            for batch_idx in range(int(sampler.total_traj_num / args.batch_size)):
                batch_data, batch_sd = sampler.next_batch(args.batch_size)
                batch_s, batch_d = batch_sd

                feed = dict(zip(model.input_form, batch_data))
                feed[model.s_inputs] = batch_s
                feed[model.d_inputs] = batch_d

                rec_loss, cate_loss, latent_loss, _ = sess.run(
                    [model.rec_loss, model.cate_loss, model.latent_loss, model.train_op], feed)
                sd_loss, _ = sess.run([model.sd_loss, model.sd_train_op], feed)

                all_loss.append([rec_loss, cate_loss, latent_loss, sd_loss])

            val_loss = compute_output(model.loss, sess, model, sampler,
                                      purpose="val", callback=np.mean)
            if len(all_val_loss) > 0 and val_loss >= all_val_loss[-1]:
                print("Early termination with val loss: {}:".format(val_loss))
                break
            all_val_loss.append(val_loss)

            end = time.time()
            print("epoch: {}\tval loss: {}\telapsed time: {}".format(
                epoch, val_loss, end - start))
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
        all_likelihood = compute_output(model.batch_likelihood, sess, model, sampler,
                                        purpose="train", callback=np.concatenate)
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
    parser.add_argument('--data_filename', type=str, default="../data/processed_porto{}.csv",
                        help='data file')
    parser.add_argument('--map_size', type=tuple, default=(51, 158),
                        help='size of map')

    # parser.add_argument('--data_filename', type=str, default="../data/processed_beijing{}.csv",
    #                     help='data file')
    # parser.add_argument('--map_size', type=tuple, default=(130, 130),
    #                     help='size of map')

    parser.add_argument('--model_type', type=str, default="",
                        help='choose a model')

    parser.add_argument('--x_latent_size', type=int, default=32,
                        help='size of input embedding')
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--mem_num', type=int, default=5,
                        help='size of sd memory')

    parser.add_argument('--neg_size', type=int, default=64,
                        help='size of negative sampling')
    parser.add_argument('--num_epochs', type=int, default=20,
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
    parser.add_argument('--pt', type=bool, default=False,
                        help='partial trajectory evaluation')

    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.eval:
        evaluate()
    elif args.pt:
        pretrain()
    else:
        train()
