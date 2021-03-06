#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco.const import *


def define_computation_graph(vocab_size: int, batch_size: int, is_training: bool=False):

    # Placeholders for input and output
    inputs = tf.placeholder(tf.int32, shape=(
        batch_size, NUM_STEPS), name='x')  # (time, batch)
    targets = tf.placeholder(tf.int32, shape=(
        batch_size, NUM_STEPS), name='y')  # (time, batch)

    with tf.name_scope('Embedding'):
        embedding = tf.get_variable(
            'word_embedding', [vocab_size, EMBEDDING_SIZE])
        input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

        if is_training and DROPOUT < 1:
            input_embeddings = tf.nn.dropout(input_embeddings, DROPOUT)

    # with tf.name_scope('simple_RNN'):
    #    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    #    initial_state = cell.zero_state(batch_size, tf.float32)
    #    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
    #        cell, input_embeddings, initial_state=initial_state)

    # with tf.name_scope('bidir_GRU_RNN'):
    #    # define forward cell (use only half of length due to bidirectionality)
    #    cell_fw = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE/2)
    #    # define backward cell (use only half of length due to bidirectionality)
    #    cell_bw = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE/2)

    #    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    #    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    #    bi_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_embeddings, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)

    #    rnn_outputs = tf.concat(bi_outputs, -1)

    with tf.name_scope('stacked_GRU_RNN'):

        def cell():
            # define GRU cell with dropout
            cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
            if is_training and DROPOUT < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=DROPOUT)
            return cell

        # stack multiple layers
        stacked_gru = tf.contrib.rnn.MultiRNNCell(
            [cell() for _ in range(NUM_LAYERS)])

        initial_state = stacked_gru.zero_state(batch_size, tf.float32)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            stacked_gru, input_embeddings, initial_state=initial_state)

    with tf.name_scope('Final_Projection'):
        w = tf.get_variable('w', shape=(HIDDEN_SIZE, vocab_size))
        b = tf.get_variable('b', vocab_size)

        def final_projection(x): return tf.matmul(x, w) + b
        logits = map_fn(final_projection, rnn_outputs)

    with tf.name_scope('Cost'):
        # weighted average cross-entropy (log-perplexity) per symbol
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=tf.ones(
                                                    [batch_size, NUM_STEPS]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return inputs, targets, loss, train_step, logits, summary
