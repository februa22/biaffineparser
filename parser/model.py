# -*- coding: utf-8 -*-
from .progress_bar import Progbar
from . import utils

from sklearn.utils import shuffle
import tensorflow as tf


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
        pass

    def create_embedding_layer(self):
        pass

    def create_lstm_layer(self):
        pass

    def create_mlp_layer(self):
        pass

    def create_biaffine_layer(self):
        #adding arc and label logits
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

#biaffine 연산 레이어
def add_biaffine_layer(input1, W, input2, device, num_outputs=1, bias_x=False, bias_y=False):
    #input의 shape을 받아옴
    batch_size, batch_len, dim = input1.shape

    if bias_x:
        input1 = tf.concat((input1, tf.ones(batch_size, batch_len, 1)), axis=2)
    if bias_y:
        input2 = tf.concat((input2, tf.ones(batch_size, batch_len, 1)), axis=2)

    nx = dim + bias_x #501
    ny = dim + bias_y #501

    W = tf.reshape(W, shape=(nx, num_outputs * W.size()[-1]))
    lin = tf.matmul(tf.reshape(input1, shape=(batch_size * batch_len, nx)), W)
    lin = tf.reshape(lin, shape=(batch_size, num_outputs * batch_len, ny))
    blin = tf.matmul(lin, tf.transpose(input2, perm=(1, 2)))
    blin = tf.reshape(blin, (batch_size, batch_len, num_outputs, batch_len))
    
    if num_outputs == 1:
        blin = tf.squeeze(blin, axis=2)
    else:
        blin = tf.transpose(blin, perm=(2,3))
    return blin