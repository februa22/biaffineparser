# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.utils import shuffle

from . import utils
from .progress_bar import Progbar


class Model(object):
    def __init__(self, hparams, word_vocab_table, pos_vocab_table, rels_vocab_table, heads_vocab_table,
                 word_embedding, pos_embedding, device='gpu'):
        print(f'word_vocab_table: {len(word_vocab_table)}')
        print(f'pos_vocab_table: {len(pos_vocab_table)}')
        print(f'rels_vocab_table: {len(rels_vocab_table)}')
        print(f'heads_vocab_table: {len(heads_vocab_table)}')
        print(f'word_embedding: {word_embedding.shape}')
        print(f'pos_embedding: {pos_embedding.shape}')
        self.hparams = hparams
        self.word_vocab_table = word_vocab_table
        self.pos_vocab_table = pos_vocab_table
        self.rels_vocab_table = rels_vocab_table
        self.heads_vocab_table = heads_vocab_table
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.build()

    def build(self):
        self.create_placeholders()
        self.create_embedding_layer()
        self.create_lstm_layer()
        self.create_mlp_layer()
        self.create_biaffine_layer()
        self.create_logits_op()
        self.create_pred_op()
        self.create_loss_op()
        self.create_train_op()
        self.create_init_op()

    def create_placeholders(self):
        # TODO(jongseong): 실제로 모델에 필요한 `placeholder`로 교체
        self.word_ids = tf.placeholder(
            tf.int32, shape=[None, None], name="word_ids")
        self.seq_len = tf.placeholder(
            tf.int32, shape=[None], name="seq_len")

    def create_embedding_layer(self):
        with tf.variable_scope('embeddings'):
            _word_embedding = tf.Variable(
                self.word_embedding, name="_word_embedding", dtype=tf.float32)
            word_embedding = tf.nn.embedding_lookup(
                _word_embedding, self.word_ids, name="word_embedding")

            _pos_embedding = tf.Variable(
                self.pos_embedding, name="_pos_embedding", dtype=tf.float32)
            pos_embedding = tf.nn.embedding_lookup(
                _pos_embedding, self.word_ids, name="pos_embedding")

            self.embeddings = tf.concat(
                [word_embedding, pos_embedding], axis=-1)

    def create_lstm_layer(self):
        with tf.variable_scope('bi-lstm'):
            self.output = add_stacked_lstm_layers(
                self.hparams, self.embeddings, self.seq_len)

    def create_mlp_layer(self):
        with tf.variable_scope('mlp'):
            s = tf.shape(self.output)
            self.output = tf.reshape(self.output, [-1, s[-1]])
            self.h_arc_head, self.h_arc_dep, self.h_label_head, self.h_label_dep = mlp_for_arc_and_label(
                self.hparams, self.output)

    def create_biaffine_layer(self):
        # adding arc and label logits
        #arc_logits = biaffine(h_arc_dep, self.W_arc, h_arc_head, self.device, num_outputs=1, bias_x=True)
        #label_logits = biaffine(h_label_dep, self.W_label, h_label_head, self.device, num_outputs=self.N_CLASSES, bias_x=True, bias_y=True)
        pass

    def create_logits_op(self):
        pass

    def create_pred_op(self):
        pass

    def create_loss_op(self):
        pass

    def create_train_op(self):
        pass

    def create_init_op(self):
        pass

    def train(self, epochs, sentences_indexed, pos_indexed, rels_indexed, heads_padded):
        print('#'*30)
        print(f'epochs {epochs}')
        print(f'sentences_indexed {sentences_indexed.shape}')
        print(f'pos_indexed {pos_indexed.shape}')
        print(f'rels_indexed {rels_indexed.shape}')
        print(f'heads_padded {heads_padded.shape}')
        print('#'*30)
        val_sentences, val_pos, val_rels, val_heads, val_maxlen = utils.get_dataset_multiindex(
            self.hparams.dev_filename)

        val_sentences_indexed = utils.get_indexed_sequences(
            val_sentences, self.word_vocab_table, val_maxlen)
        val_pos_indexed = utils.get_indexed_sequences(
            val_pos, self.pos_vocab_table, val_maxlen)
        val_rels_indexed = utils.get_indexed_sequences(
            val_rels, self.rels_vocab_table, val_maxlen)
        val_heads_padded = utils.get_indexed_sequences(
            val_heads, self.heads_vocab_table, val_maxlen, just_pad=True)

        for epoch in range(epochs):
            model_loss = []
            heads_acc = []
            rels_acc = []
            val_model_loss = []
            val_heads_acc = []
            val_rels_acc = []
            test_model_loss = []
            test_heads_acc = []
            test_rels_acc = []
            # reset progbar each epoch
            progbar = Progbar(len(sentences_indexed))
            sentences_indexed, pos_indexed, rels_indexed, heads_padded = shuffle(sentences_indexed, pos_indexed,
                                                                                 rels_indexed,
                                                                                 heads_padded)
            for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
                    sentences_indexed, pos_indexed, rels_indexed, heads_padded, batch_size=100):
                break
            break


def add_stacked_lstm_layers(hparams, word_embedding, lengths):
    cell = tf.contrib.rnn.LSTMCell
    cells_fw = [cell(hparams.lstm_hidden_size) for _ in range(hparams.num_lstm_layers)]
    cells_bw = [cell(hparams.lstm_hidden_size) for _ in range(hparams.num_lstm_layers)]
    if hparams.dropout > 0.0:
        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
        cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=word_embedding,
        sequence_length=lengths,
        dtype=tf.float32,
        scope="bi-lstm")
    return outputs


def create_weight_and_bias(n_input, n_output):
    weights = {
        'w1': tf.get_variable('w1', shape=[n_input, n_output], dtype=tf.float32),
        'w2': tf.get_variable('w2', shape=[n_output, n_output], dtype=tf.float32),
    }
    biases = {
        'b1': tf.get_variable('b1', shape=[n_output], dtype=tf.float32, initializer=tf.zeros_initializer()),
        'b2': tf.get_variable('b2', shape=[n_output], dtype=tf.float32, initializer=tf.zeros_initializer()),
    }
    return weights, biases


def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    return layer_2


def mlp_with_scope(x, n_input, n_output, scope):
    with tf.variable_scope(scope):
        weights, biases = create_weight_and_bias(n_input, n_output)
        h = multilayer_perceptron(x, weights, biases)
    return h


def mlp_for_arc(hparams, x):
    h_arc_head = mlp_with_scope(
        x, 2*hparams.lstm_hidden_size, hparams.arc_mlp_units, 'arc_head')
    h_arc_dep = mlp_with_scope(
        x, 2*hparams.lstm_hidden_size, hparams.arc_mlp_units, 'arc_dep')
    return h_arc_head, h_arc_dep


def mlp_for_label(hparams, x):
    h_label_head = mlp_with_scope(
        x, 2*hparams.lstm_hidden_size, hparams.label_mlp_units, 'label_head')
    h_label_dep = mlp_with_scope(
        x, 2*hparams.lstm_hidden_size, hparams.label_mlp_units, 'label_dep')
    return h_label_head, h_label_dep


def mlp_for_arc_and_label(hparams, x):
    x = tf.nn.dropout(x, hparams.mlp_dropout)
    h_arc_head, h_arc_dep = mlp_for_arc(hparams, x)
    h_label_head, h_label_dep = mlp_for_label(hparams, x)
    return h_arc_head, h_arc_dep, h_label_head, h_label_dep


def add_biaffine_layer(input1, W, input2, device, num_outputs=1, bias_x=False, bias_y=False):
    """ biaffine 연산 레이어 """
    # input의 shape을 받아옴
    batch_size, batch_len, dim = input1.shape

    if bias_x:
        input1 = tf.concat((input1, tf.ones(batch_size, batch_len, 1)), axis=2)
    if bias_y:
        input2 = tf.concat((input2, tf.ones(batch_size, batch_len, 1)), axis=2)

    nx = dim + bias_x  # 501
    ny = dim + bias_y  # 501

    W = tf.reshape(W, shape=(nx, num_outputs * W.size()[-1]))
    lin = tf.matmul(tf.reshape(input1, shape=(batch_size * batch_len, nx)), W)
    lin = tf.reshape(lin, shape=(batch_size, num_outputs * batch_len, ny))
    blin = tf.matmul(lin, tf.transpose(input2, perm=(1, 2)))
    blin = tf.reshape(blin, (batch_size, batch_len, num_outputs, batch_len))

    if num_outputs == 1:
        blin = tf.squeeze(blin, axis=2)
    else:
        blin = tf.transpose(blin, perm=(2, 3))
    return blin
