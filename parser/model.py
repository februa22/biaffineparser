# -*- coding: utf-8 -*-
import pdb

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from . import utils

class Model(object):
    def __init__(self, hparams, word_vocab_table, char_vocab_table, pos_vocab_table,
                 rels_vocab_table, heads_vocab_table, word_embedding, char_embedding, pos_embedding):
        print('#'*30)
        print(f'word_vocab_table: {len(word_vocab_table)}')
        print(f'char_vocab_table: {len(char_vocab_table)}')
        print(f'pos_vocab_table: {len(pos_vocab_table)}')
        print(f'rels_vocab_table: {len(rels_vocab_table)}')
        print(f'heads_vocab_table: {len(heads_vocab_table)}')
        print(f'word_embedding: {word_embedding.shape}')
        print(f'char_embedding: {char_embedding.shape}')
        print(f'pos_embedding: {pos_embedding.shape}')
        print('#'*30)
        self.word_vocab_table = word_vocab_table
        self.char_vocab_table = char_vocab_table
        self.pos_vocab_table = pos_vocab_table
        self.rels_vocab_table = rels_vocab_table
        self.heads_vocab_table = heads_vocab_table
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.pos_embedding = pos_embedding

        self.hparams = hparams
        self.n_classes = len(rels_vocab_table)

        self.word_pad_id = word_vocab_table[utils.GLOBAL_PAD_SYMBOL]
        self.char_pad_id = char_vocab_table[utils.GLOBAL_PAD_SYMBOL]
        self.head_pad_id = tf.constant(
            heads_vocab_table[utils.GLOBAL_PAD_SYMBOL])
        self.rel_pad_id = tf.constant(
            rels_vocab_table[utils.GLOBAL_PAD_SYMBOL])

        self.embed_dropout = 0.0
        self.lstm_dropout = 0.0
        self.mlp_dropout = 0.0

        self.train_mode = tf.contrib.learn.ModeKeys.TRAIN
        self.eval_mode = tf.contrib.learn.ModeKeys.EVAL
        self.infer_mode = tf.contrib.learn.ModeKeys.INFER

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self._build_graph()

    def _build_graph(self):
        self.create_placeholders()
        self.create_embedding_layer()
        self.create_lstm_layer()
        self.create_mlp_layer()
        self.create_biaffine_layer()
        self.create_loss_op()
        self.create_train_op()
        self.create_uas_and_las_op()

    def build(self):
        # Check out_dir
        if not tf.gfile.Exists(self.hparams.out_dir):
            utils.print_out(
                f"# Creating output directory {self.hparams.out_dir} ...")
            tf.gfile.MakeDirs(self.hparams.out_dir)

        self.initializer = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        config_proto = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
        config_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_proto)
        # use debug mode if debug is True
        if self.hparams.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(self.initializer)
        self.merge_summaries_and_create_writer(self.sess)

    def new_sess_and_restore(self, save_path):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path)

    def create_placeholders(self):
        # word_ids, pos_ids shape: (?,?,?) => (batch_size, sequence_words, word_morphs)
        # 한국/NNP
        self.word_ids = tf.placeholder(
            tf.int32, shape=[None, None, None], name='word_ids')

        # 한국
        self.char_ids = tf.placeholder(
            tf.int32, shape=[None, None, None], name='char_ids')

        # NNP
        self.pos_ids = tf.placeholder(
            tf.int32, shape=[None, None, None], name='pos_ids')

        self.head_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='head_ids')
        self.rel_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='rel_ids')
        self.sequence_length = tf.placeholder(
            tf.int32, shape=[None], name='sequence_length')
        self.word_length = tf.placeholder(
            tf.int32, shape=[None, None], name='word_length')
        self.char_length = tf.placeholder(
            tf.int32, shape=[None, None], name='char_length')

    def create_embedding_layer(self):
        with tf.device('/cpu:0'), tf.variable_scope('embeddings'):
            # get word length
            sequence_length = tf.reshape(self.word_length, [-1])

            with tf.variable_scope('word'):
                trainable = False if self.hparams.word_embed_file else True
                _word_embedding = tf.Variable(
                    self.word_embedding, trainable=trainable,
                    name="_word_embedding", dtype=tf.float32)
                word_embedding = tf.nn.embedding_lookup(
                    _word_embedding, self.word_ids, name="word_embedding")

            with tf.variable_scope('pos'):
                trainable = False if self.hparams.pos_embed_file else True
                _pos_embedding = tf.Variable(
                    self.pos_embedding, trainable=trainable,
                    name="_pos_embedding", dtype=tf.float32)
                pos_embedding = tf.nn.embedding_lookup(
                    _pos_embedding, self.pos_ids, name="pos_embedding")

            word_embedding = tf.concat([word_embedding, pos_embedding], axis=-1)
            shape = tf.shape(self.word_ids)
            dim = self.hparams.word_embed_size + self.hparams.pos_embed_size
            word_embedding = tf.reshape(
                word_embedding, [-1, shape[2], dim])
            word_embedding = bilstm_layer(
                word_embedding, sequence_length,
                int(dim / 2))
            word_embedding = tf.reshape(
                word_embedding, [-1, shape[1], dim])
            if self.embed_dropout > 0.0:
                keep_prob = 1.0 - self.embed_dropout
                word_embedding = tf.nn.dropout(word_embedding, keep_prob)

            with tf.variable_scope('char'):
                trainable = False if self.hparams.char_embed_file else True
                _char_embedding = tf.Variable(
                    self.char_embedding, trainable=trainable,
                    name="_char_embedding", dtype=tf.float32)
                char_embedding = tf.nn.embedding_lookup(
                    _char_embedding, self.char_ids, name="char_embedding")
                shape = tf.shape(char_embedding)
                sequence_length = tf.reshape(self.char_length, [-1])
                dim = self.hparams.char_embed_size
                # BEFORE
                char_embedding = tf.reshape(char_embedding, [-1, shape[2], dim])
                char_embedding = bilstm_layer(
                    char_embedding, sequence_length, int(dim / 2))
                char_embedding = tf.reshape(char_embedding, [-1, shape[1], dim])
                if self.embed_dropout > 0.0:
                    keep_prob = 1.0 - self.embed_dropout
                    char_embedding = tf.nn.dropout(char_embedding, keep_prob)
                # #AFTER
                # char_embedding = tf.reshape(
                #     char_embedding, [-1, shape[2], self.hparams.char_embed_size])
                # char_embedding = bilstm_layer(char_embedding, sequence_length, 100) #100
                # char_embedding = tf.reshape(
                #     char_embedding, [-1, shape[1], 200]) #200
                # if self.embed_dropout > 0.0:
                #     keep_prob = 1.0 - self.embed_dropout
                #     char_embedding = tf.nn.dropout(
                #         char_embedding, keep_prob)

            # concat
            self.embeddings = tf.concat(
                [word_embedding, char_embedding], axis=-1)

    def create_lstm_layer(self):
        with tf.variable_scope('bi-lstm'):
            self.output = add_stacked_lstm_layers(
                self.hparams, self.embeddings, self.sequence_length, self.lstm_dropout)

    def create_mlp_layer(self):
        with tf.variable_scope('mlp'):
            self.mlp_out_size = 500
            batch_size = tf.shape(self.word_ids)[0]
            self.output = tf.reshape(
                self.output, [-1, 2*self.hparams.num_lstm_units])
            # MLP
            #   h_arc_head: [batch_size * seq_len, dim]
            self.h_arc_head, self.h_arc_dep, self.h_label_head, self.h_label_dep = mlp_for_arc_and_label(
                self.hparams, self.output, self.mlp_dropout)
            # self.h_label_head2 = mlp_with_scope(
            #     self.output, 2*self.hparams.num_lstm_units, self.hparams.arc_mlp_units, self.mlp_dropout, 'label_head2')
            # Reshape
            #   h_arc_head: [batch_size, seq_len, dim])
            self.h_arc_head = tf.reshape(
                self.h_arc_head, [batch_size, -1, self.mlp_out_size])
            self.h_arc_dep = tf.reshape(
                self.h_arc_dep, [batch_size, -1, self.mlp_out_size])
            self.h_label_head = tf.reshape(
                self.h_label_head, [batch_size, -1, self.mlp_out_size])
            self.h_label_dep = tf.reshape(
                self.h_label_dep, [batch_size, -1, self.mlp_out_size])
            # self.h_label_head2 = tf.reshape(
            #     self.h_label_head2, [batch_size, -1, self.hparams.arc_mlp_units])

    def create_biaffine_layer(self):
        """ adding arc and label logits """
        # logit for arc and label
        with tf.variable_scope('arc'):
            W_arc = tf.get_variable('w_arc', [self.mlp_out_size + 1, 1, self.mlp_out_size],
                                    dtype=tf.float32, initializer=tf.orthogonal_initializer)
            arc_logits = add_biaffine_layer(
                self.h_arc_dep, W_arc, self.h_arc_head, self.hparams.device, num_outputs=1, bias_x=True, bias_y=False)

            W_arc2 = tf.get_variable('w_arc2', [self.mlp_out_size + 1, 1, self.mlp_out_size],
                                    dtype=tf.float32, initializer=tf.orthogonal_initializer)
            arc_logits2 = add_biaffine_layer(
                self.h_label_head, W_arc2, self.h_arc_head, self.hparams.device, num_outputs=1, bias_x=True, bias_y=False)

            W_arc3 = tf.get_variable('w_arc3', [self.mlp_out_size + 1, 1, self.mlp_out_size],
                                    dtype=tf.float32, initializer=tf.orthogonal_initializer)
            arc_logits3 = add_biaffine_layer(
                self.h_label_dep, W_arc3, self.h_arc_head, self.hparams.device, num_outputs=1, bias_x=True, bias_y=False)

            self.arc_logits = tf.add(
                tf.divide(arc_logits, 3),
                tf.divide(arc_logits2, 3) + tf.divide(arc_logits3, 3),
                name='arc_logits')

        with tf.variable_scope('label'):
            W_label = tf.get_variable('w_label', [self.mlp_out_size + 1, self.n_classes,
                                                  self.mlp_out_size + 1], dtype=tf.float32, initializer=tf.orthogonal_initializer)
            full_label_logits = add_biaffine_layer(self.h_label_dep, W_label, self.h_label_head,
                                                   self.hparams.device, num_outputs=self.n_classes, bias_x=True, bias_y=True)  # [batch,seq_length,heads,label_classes]

            # turn off the padding tensor to false
            gold_heads = self.head_ids
            mask = tf.cast(tf.not_equal(
                gold_heads, self.head_pad_id), dtype=tf.int32)
            # convert to zero for padding values => [batch, sent_len]
            pred_arcs = tf.multiply(gold_heads, mask)

            # Gather label logits from predicted or gold heads with gather
            # need to compute batch_size, seq_length of pred_arcs(head_ids) for getting indices
            pred_arcs_shape = tf.shape(pred_arcs)
            batch_size = pred_arcs_shape[0]
            seq_len = pred_arcs_shape[1]
            batch_idx = tf.range(0, batch_size)  # [0, 1]
            batch_idx = tf.expand_dims(batch_idx, 1)  # [[0], [1]]
            batch_idx = tf.tile(batch_idx, [1, seq_len])  # [[0, 0], [1, 1]]
            seq_idx = tf.range(0, seq_len)  # [0, 1]
            seq_idx = tf.expand_dims(seq_idx, 0)  # [[0, 1]]
            seq_idx = tf.tile(seq_idx, [batch_size, 1])  # [[0, 1], [0, 1]]
            # [[batch_idx, seq_idx, head_idx], ...]
            indices = tf.stack([batch_idx, seq_idx, pred_arcs], 2)
            self.label_logits = tf.gather_nd(
                full_label_logits, indices=indices, name='label_logits')

    # compute loss
    def compute_loss(self, logits, gold_labels, sequence_length):
        # computing loss for labels
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=gold_labels)
        mask = tf.sequence_mask(sequence_length)
        # slice loss and mask for getting rid of root_word => [:, 1:]
        masked_loss = tf.boolean_mask(loss[:, 1:], mask[:, 1:])
        mean_loss = tf.reduce_mean(masked_loss)
        return mean_loss

    # loss logit
    def create_loss_op(self):
        max_len = tf.reduce_max(self.sequence_length)
        gold_heads = tf.one_hot(self.head_ids, max_len)
        loss_heads = self.compute_loss(
            self.arc_logits, gold_heads, self.sequence_length)
        gold_rels = tf.one_hot(self.rel_ids, self.n_classes)
        loss_rels = self.compute_loss(
            self.label_logits, gold_rels, self.sequence_length)
        self.train_loss = loss_heads + loss_rels

    def create_train_op(self):
        if self.hparams.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.hparams.learning_rate)
        elif self.hparams.optimizer == 'adam':
            assert float(
                self.hparams.learning_rate
            ) <= 0.001, f'! High Adam learning rate {self.hparams.learning_rate}'
            opt = tf.train.AdamOptimizer(
                learning_rate=self.hparams.learning_rate, beta2=self.hparams.decay_factor)
        else:
            raise ValueError(f'Unknown optimizer {self.hparams.optimizer}')
        self.update = opt.minimize(
            self.train_loss, global_step=self.global_step)

    def create_uas_and_las_op(self):
        """ UAS and LAS"""
        with tf.variable_scope('uas'):
            sequence_mask = tf.sequence_mask(self.sequence_length)
            self.head_preds = tf.argmax(
                self.arc_logits, axis=-1, output_type=tf.int32)
            self.masked_head_preds = tf.boolean_mask(
                self.head_preds[:, 1:], sequence_mask[:, 1:])
            masked_head_ids = tf.boolean_mask(
                self.head_ids[:, 1:], sequence_mask[:, 1:])
            head_correct = tf.equal(self.masked_head_preds, masked_head_ids)
            self.uas = tf.reduce_mean(tf.cast(head_correct, tf.float32))

        with tf.variable_scope('las'):
            self.rel_preds = tf.argmax(
                self.label_logits, axis=-1, output_type=tf.int32)
            self.masked_rel_preds = tf.boolean_mask(
                self.rel_preds[:, 1:], sequence_mask[:, 1:])
            masked_rel_ids = tf.boolean_mask(
                self.rel_ids[:, 1:], sequence_mask[:, 1:])
            rel_correct = tf.equal(self.masked_rel_preds, masked_rel_ids)
            head_rel_correct = tf.logical_and(head_correct, rel_correct)
            self.las = tf.reduce_mean(tf.cast(head_rel_correct, tf.float32))

    def merge_summaries_and_create_writer(self, sess):
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(
            self.hparams.out_dir, sess.graph)

    def add_summary(self, global_step, tag, value):
        """Add a new summary to the current summary_writer.
        Useful to log things that are not part of the training graph, e.g., tag=UAS.
        """
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step)

    def _run_session(self, sentences_indexed, chars_indexed, pos_indexed,
                     heads_indexed=None, rels_indexed=None):
        self.embed_dropout = self.hparams.embed_dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        self.lstm_dropout = self.hparams.lstm_dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        self.mlp_dropout = self.hparams.mlp_dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        sequence_length = utils.get_sequence_length(
            sentences_indexed, self.word_pad_id)

        max_seq_len = utils.get_max(sequence_length)
        sentences_indexed = sentences_indexed[:, :max_seq_len, :]
        chars_indexed = chars_indexed[:, :max_seq_len]
        pos_indexed = pos_indexed[:, :max_seq_len, :]

        if heads_indexed is not None:
            heads_indexed = heads_indexed[:, :max_seq_len]
        if rels_indexed is not None:
            rels_indexed = rels_indexed[:, :max_seq_len]

        word_length = utils.get_word_length(
            sentences_indexed, self.word_pad_id)
        max_word_len = utils.get_max(word_length)
        sentences_indexed = sentences_indexed[:, :, :max_word_len]
        pos_indexed = pos_indexed[:, :, :max_word_len]

        char_length = utils.get_word_length(chars_indexed, self.char_pad_id)
        max_char_len = utils.get_max(char_length)
        chars_indexed = chars_indexed[:, :, :max_char_len]

        feed_dict = {
            self.word_ids: sentences_indexed,
            self.char_ids: chars_indexed,
            self.pos_ids: pos_indexed,
            self.sequence_length: sequence_length,
            self.word_length: word_length,
            self.char_length: char_length,
        }

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            feed_dict[self.head_ids] = heads_indexed
            feed_dict[self.rel_ids] = rels_indexed
            fetches = [self.update, self.train_loss,
                       self.uas, self.las, self.global_step]
            return self.sess.run(fetches, feed_dict)
            # utils.print_out(f'\n # loss={loss}, uas={uas}, las={las}, global_step={global_step}')
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            feed_dict[self.head_ids] = heads_indexed
            feed_dict[self.rel_ids] = rels_indexed
            fetches = [self.train_loss, self.uas, self.las,
                       self.masked_head_preds, self.masked_rel_preds,
                       self.sequence_length]
            return self.sess.run(fetches, feed_dict)
            # utils.print_out(f'\n # loss={loss}, uas={uas}, las={las}, global_step={global_step}')
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            fetches = [self.masked_head_preds, self.masked_rel_preds]
            return self.sess.run(fetches, feed_dict)

    def train_step(self, data):
        self.mode = self.train_mode
        (sentences_indexed, chars_indexed,
         pos_indexed, heads_indexed, rels_indexed) = data
        return self._run_session(sentences_indexed, chars_indexed, pos_indexed,
                                 heads_indexed, rels_indexed)

    def eval_step(self, data):
        self.mode = self.eval_mode
        (sentences_indexed, chars_indexed,
         pos_indexed, heads_indexed, rels_indexed) = data
        return self._run_session(sentences_indexed, chars_indexed, pos_indexed,
                                 heads_indexed, rels_indexed)

    def inference_step(self, data):
        self.mode = self.infer_mode
        (sentences_indexed, chars_indexed, pos_indexed) = data
        return self._run_session(sentences_indexed, chars_indexed, pos_indexed)

    def save(self, save_path):
        self.saver.save(self.sess, save_path)


def add_stacked_lstm_layers(hparams, word_embedding, lengths, dropout):
    cell = tf.contrib.rnn.LSTMCell
    cells_fw = [cell(hparams.num_lstm_units)
                for _ in range(hparams.num_lstm_layers)]
    cells_bw = [cell(hparams.num_lstm_units)
                for _ in range(hparams.num_lstm_layers)]
    if dropout > 0.0:
        keep_prob = 1.0 - dropout
        cells_fw = [tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=keep_prob,
            state_keep_prob=keep_prob) for cell in cells_fw]
        cells_bw = [tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=keep_prob,
            state_keep_prob=keep_prob) for cell in cells_bw]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=word_embedding,
        sequence_length=lengths,
        dtype=tf.float32)
    return outputs


def create_weight_and_bias(n_input, n_output):
    weights = {
        'w1': tf.get_variable('w1', shape=[n_input, n_output], dtype=tf.float32),
        'w2': tf.get_variable('w2', shape=[n_output, 500], dtype=tf.float32)
    }
    biases = {
        'b1': tf.get_variable('b1', shape=[n_output], dtype=tf.float32, initializer=tf.zeros_initializer()),
        'b2': tf.get_variable('b2', shape=[500], dtype=tf.float32, initializer=tf.zeros_initializer())
    }
    return weights, biases


def multilayer_perceptron(x, weights, biases, dropout):
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    if dropout > 0.0:
        keep_prob = 1.0 - dropout
        out_layer = tf.nn.dropout(out_layer, keep_prob)
    return out_layer


def mlp_with_scope(x, n_input, n_output, dropout, scope):
    with tf.variable_scope(scope):
        weights, biases = create_weight_and_bias(n_input, n_output)
        h = multilayer_perceptron(x, weights, biases, dropout)
    return h


def mlp_for_arc(hparams, x, dropout):
    h_arc_head = mlp_with_scope(
        x, 2*hparams.num_lstm_units, hparams.arc_mlp_units, dropout, 'arc_head')
    h_arc_dep = mlp_with_scope(
        x, 2*hparams.num_lstm_units, hparams.arc_mlp_units, dropout, 'arc_dep')
    return h_arc_head, h_arc_dep


def mlp_for_label(hparams, x, dropout):
    h_label_head = mlp_with_scope(
        x, 2*hparams.num_lstm_units, hparams.label_mlp_units, dropout, 'label_head')
    h_label_dep = mlp_with_scope(
        x, 2*hparams.num_lstm_units, hparams.label_mlp_units, dropout, 'label_dep')
    return h_label_head, h_label_dep


def mlp_for_arc_and_label(hparams, x, dropout):
    h_arc_head, h_arc_dep = mlp_for_arc(hparams, x, dropout)
    h_label_head, h_label_dep = mlp_for_label(hparams, x, dropout)
    return h_arc_head, h_arc_dep, h_label_head, h_label_dep


def add_biaffine_layer(input1, W, input2, device, num_outputs=1, bias_x=False, bias_y=False):
    """ biaffine 연산 레이어 """
    # input의 shape을 받아옴
    input1_shape = tf.shape(input1)
    batch_size = input1_shape[0]
    batch_len = input1_shape[1]
    dim = input1_shape[2]

    if bias_x:
        input1 = tf.concat(
            [input1, tf.ones([batch_size, batch_len, 1])], axis=2)
    if bias_y:
        input2 = tf.concat(
            [input2, tf.ones([batch_size, batch_len, 1])], axis=2)

    nx = dim + bias_x  # 501
    ny = dim + bias_y  # 501

    W = tf.reshape(W, shape=(nx, num_outputs * tf.shape(W)[-1]))
    lin = tf.matmul(tf.reshape(input1, shape=(batch_size * batch_len, nx)), W)
    lin = tf.reshape(lin, shape=(batch_size, num_outputs * batch_len, ny))
    blin = tf.matmul(lin, tf.transpose(input2, perm=[0, 2, 1]))
    blin = tf.reshape(blin, (batch_size, batch_len, num_outputs, batch_len))

    if num_outputs == 1:
        blin = tf.squeeze(blin, axis=2)
    else:
        blin = tf.transpose(blin, perm=[0, 1, 3, 2])
    return blin


def bilstm_layer(inputs, sequence_length, num_units):
    cell = tf.contrib.rnn.LSTMCell
    cell_fw = cell(num_units, state_is_tuple=True)
    cell_bw = cell(num_units, state_is_tuple=True)
    _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw,
        inputs, sequence_length=sequence_length,
        dtype=tf.float32)
    final_states = tf.concat([output_fw, output_bw], axis=-1)
    return final_states
