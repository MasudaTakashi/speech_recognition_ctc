from __future__ import division
from __future__ import print_function

import os
import numpy as np
import chainer
import tensorflow as tf

from utils import get_args, print_progress, BatchIterator, get_shape, ResultWriter

n_label = 48
blank_symbol = n_label
data_dir = "data/timit"

n_input_units = (40+1)*3
n_hidden_units = 256
n_output_units = n_label+1 # phonemes & blank_symbol


def _inference(input_ph, seq_lengths_ph):
    with tf.name_scope("inference") as scope:
        weight_in = tf.Variable(tf.truncated_normal([n_input_units, n_hidden_units],
                                                    stddev=np.sqrt(2.0 / (n_input_units + n_hidden_units))),
                                name="weight_in")
        bias_in = tf.zeros([n_hidden_units], name="bias_in")
        weight_out = tf.Variable(tf.truncated_normal([n_hidden_units, n_output_units],
                                                     stddev=np.sqrt(2.0 / (n_hidden_units + n_output_units))),
                                name="weight_out")
        bias_out = tf.zeros([n_output_units], name="bias_out")

        (batchsize, seq_length, n_in) = get_shape(input_ph)
        in1 = tf.transpose(input_ph, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, n_input_units])
        in3 = tf.matmul(in2, weight_in) + bias_in
        in4 = tf.split(0, seq_length, in3)
        # in4: list of tensor (shape=[batchsize, n_hidden_units]), the length of in4 is seq_length

        f_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, state_is_tuple=True)
        #b_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, state_is_tuple=True)
        #todo: dynamic_rnn(cell, in4, time_major=True)
        rnn_output, _ = tf.nn.rnn(f_cell, in4, dtype=tf.float32, sequence_length=seq_lengths_ph)
        #rnn_output, _, _ = tf.nn.bidirectional_rnn(f_cell, b_cell, in4, dtype=tf.float32, sequence_length=seq_lengths_ph)
        # rnn_output: list of tensor (shape=[batchsize, n_hidden_units]), the length of in4 is seq_length
        out1 = tf.reshape(tf.concat(1, rnn_output), [-1, n_hidden_units])
        out2 = tf.matmul(out1, weight_out) + bias_out
        out3 = tf.reshape(out2, [-1, seq_length, n_output_units])
        out4 = tf.transpose(out3, [1, 0, 2])
        output_op = out4

        return output_op


def _loss(logits, labels_ph, seq_lengths_ph):
    ctc_loss = tf.nn.ctc_loss(logits, labels_ph, seq_lengths_ph)
    loss_op = tf.reduce_mean(ctc_loss)
    return loss_op


def _vis_loss(loss_op, seq_lengths):
    #todo: fix me. not use loss_op (already use tf.reduce_mean)
    loss_per_frame = loss_op / tf.to_float(seq_lengths)
    return loss_per_frame


def _error(logits, labels, seq_lengths_ph):
    predictions = tf.nn.ctc_beam_search_decoder(logits, seq_lengths_ph, beam_width=10, top_paths=1)[0][0]
    errors = tf.edit_distance(tf.to_int32(predictions), labels, normalize=False)
    error_op = tf.reduce_sum(errors) / tf.to_float(tf.size(labels.values))
    return error_op


def main():
    args = get_args()
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print()


    # Load dataset
    print('Load dataset')
    train_iter = BatchIterator(args.batchsize, os.path.join(data_dir, "train"))
    dev_iter = BatchIterator(args.batchsize, os.path.join(data_dir, "dev"))
    test_iter = BatchIterator(args.batchsize, os.path.join(data_dir, "test"))
    max_length = max(
            train_iter.get_maxlen(),
            dev_iter.get_maxlen(),
            test_iter.get_maxlen()
    )
    train_iter.set_maxlen(max_length)
    dev_iter.set_maxlen(max_length)
    test_iter.set_maxlen(max_length)


    # Define graph
    print('Defining graph')
    graph = tf.Graph()
    optimizer = tf.train.AdamOptimizer()
    with graph.as_default():
        input_ph = tf.placeholder(tf.float32, shape=[None, max_length, n_input_units])
        labels_indices = tf.placeholder(tf.int64)
        labels_values = tf.placeholder(tf.int32)
        labels_shape = tf.placeholder(tf.int64)
        labels = tf.SparseTensor(labels_indices, labels_values, labels_shape)
        seq_lengths_ph = tf.placeholder(tf.int32)

        output_op = _inference(input_ph, seq_lengths_ph)
        loss_op = _loss(output_op, labels, seq_lengths_ph)
        #vis_loss_op = _vis_loss(loss_op, seq_lengths_ph)
        train_op = optimizer.minimize(loss_op)
        error_op = _error(output_op, labels, seq_lengths_ph)

    with tf.Session(graph=graph) as session:
        print('initializing')
        tf.initialize_all_variables().run()
        headers = ["epoch", "train_loss", "dev_loss", "dev_PER"]
        result_writer = ResultWriter(headers)
        for epoch in xrange(args.epoch):
            print('-' * 50)
            print('Epoch', epoch+1, '...')
            # train
            train_losses = []
            progress = 0
            for x_batches, t_batches, seq_lengths in train_iter:
                feed_dict = {
                    input_ph: x_batches,
                    labels_indices: t_batches[0],
                    labels_values: t_batches[1],
                    labels_shape: t_batches[2],
                    seq_lengths_ph: seq_lengths,
                }
                loss, _ = session.run([loss_op, train_op], feed_dict=feed_dict)
                train_losses.append(loss)
                progress += len(x_batches)
                print_progress(train_iter.size(), progress)
            print()
            train_loss = np.average(train_losses)
            print("train loss: {0}".format(train_loss))

            # dev
            dev_losses = []
            dev_errors = []
            progress = 0
            for x_batches, t_batches, seq_lengths in dev_iter:
                feedDict = {
                    input_ph: x_batches,
                    labels_indices: t_batches[0],
                    labels_values: t_batches[1],
                    labels_shape: t_batches[2],
                    seq_lengths_ph: seq_lengths,
                }
                loss, error = session.run([loss_op, error_op], feed_dict=feedDict)
                dev_losses.append(loss)
                dev_errors.append(error)
                progress += len(x_batches)
                print_progress(dev_iter.size(), progress)
            print()
            dev_loss = np.average(dev_losses)
            dev_error = np.average(dev_errors)
            print("dev loss: {0}, dev FER: {1}".format(dev_loss, dev_error))
            result_writer.write(epoch+1, train_loss, dev_loss, dev_error)
        result_writer.close()

if __name__ == '__main__':
    main()