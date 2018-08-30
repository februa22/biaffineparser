# -*- coding: utf-8 -*-
""" TensorFlow Deep Biaffine Attention model implementation. """
import argparse
import pdb

import numpy as np
from sklearn.utils import shuffle

from . import utils
from .model import Model
from .progress_bar import Progbar

FLAGS = None


def add_arguments(parser):
    """ Build ArgumentParser. """
    parser.register("type", "bool", lambda v: v.lower() == "true")
    #parser.register("type", "bool", lambda v: v.lower() == "false")

    # network
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Embedding size')
    parser.add_argument('--embedding_train_size', type=int, default=100,
                        help='Embedding size for train')
    parser.add_argument('--lstm_hidden_size', type=int, default=400,
                        help='Number of hidden units for LSTM')
    parser.add_argument('--pos_embedding_size', type=int, default=100,
                        help='Size of POS embedding')
    parser.add_argument('--arc_mlp_units', type=int, default=500,
                        help='Number of hidden units for MLP of arc')
    parser.add_argument('--label_mlp_units', type=int, default=100,
                        help='Number of hidden units for MLP of label')
    parser.add_argument('--dropout', type=float, default=.33,
                        help='Dropout rate')
    parser.add_argument('--embedding_dropout', type=float, default=.33,
                        help='Dropout rate for embedding')
    parser.add_argument('--mlp_dropout', type=float, default=.33,
                        help='Dropout rate for MLP')
    parser.add_argument('--num_lstm_layers', type=int, default=3,
                        help='Number of LSTM layers')

    # optimizer
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--decay_factor", type=float, default=0.9,
                        help="How much we decay.")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Num epochs to train.")

    # data
    parser.add_argument('--train_filename', type=str,
                        help='Path of train dataset')
    parser.add_argument('--dev_filename', type=str,
                        help='Path of dev dataset')
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    # Default settings works well (rarely need to change)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")

    # Misc
    parser.add_argument('--device', type=str,
                        help='Device to use')

    # Debug
    parser.add_argument('--debug', type=str2bool,
                        help='Use debugger to track down bad values during training', default=True)


def str2bool(v):
    """ for parsing bool values """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(flags):
    print('Loading dataset..')
    (sentences_indexed,  # (12543, 160)
        pos_indexed,  # (12543, 160)
        heads_padded,   # (12543, 160)
        rels_indexed,   # (12543, 160)
        words_dict,  # 400005
        pos_features_dict,  # 20
        heads_features_dict,  # 132
        rels_features_dict,  # 53
        word_embedding,  # (400005, 100)
        pos_embedding,  # (20, 100)
        maxlen,  # 160
     ) = utils.load_dataset(flags.train_filename)

    print('#'*30)
    print(f'sentences_indexed {sentences_indexed.shape}')
    print(f'pos_indexed {pos_indexed.shape}')
    print(f'rels_indexed {rels_indexed.shape}')
    print(f'heads_padded {heads_padded.shape}')
    print('#'*30)

    # sanity check
    for i, h in enumerate(heads_padded):
        if len(h) != maxlen:
            print(h)
            print('ERROR IN PADDING HEADS')
        if len(rels_indexed[i]) != maxlen:
            print(rels_indexed[i])
            print('ERROR IN PADDING RELATIONS')
        if len(np.not_equal(rels_indexed[i], rels_features_dict['<PAD>'])) != len(
                np.not_equal(h, heads_features_dict['<PAD>'])):
            print('ERROR IN LEN OF')
            print('rels', rels_indexed[i])
            print('heads', h)

    rev_vocab_words = {w: i for i, w in (words_dict.items())}
    rev_vocab_rels = {w: i for i, w in (rels_features_dict.items())}

    val_sentences, val_pos, val_rels, val_heads, val_maxlen = utils.get_dataset_multiindex(
        flags.dev_filename)
    val_sentences_indexed = utils.get_indexed_sequences(
        val_sentences, words_dict, val_maxlen)
    val_pos_indexed = utils.get_indexed_sequences(
        val_pos, pos_features_dict, val_maxlen)
    val_rels_indexed = utils.get_indexed_sequences(
        val_rels, rels_features_dict, val_maxlen)
    val_heads_padded = utils.get_indexed_sequences(
        val_heads, heads_features_dict, val_maxlen, just_pad=True)

    best_val_loss = 100
    stopcount = 0

    model = Model(flags, words_dict, pos_features_dict, rels_features_dict, heads_features_dict,
                  word_embedding, pos_embedding)

    for epoch in range(flags.num_train_epochs):
        # reset progbar each epoch
        progbar = Progbar(len(sentences_indexed))
        sentences_indexed, pos_indexed, rels_indexed, heads_padded = shuffle(
            sentences_indexed, pos_indexed, rels_indexed, heads_padded, random_state=0)
        # iterate over the train-set
        for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
                sentences_indexed, pos_indexed, rels_indexed, heads_padded, batch_size=flags.batch_size):
            uas, las = model.train_step(
                sentences_indexed_batch, pos_indexed_batch, heads_indexed_batch, rels_indexed_batch)
            break
        # iterate over the dev-set
        for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
                val_sentences_indexed, val_pos_indexed, val_rels_indexed, val_heads_padded,
                batch_size=flags.batch_size):
            uas, las = model.eval_step(
                sentences_indexed_batch, pos_indexed_batch, heads_indexed_batch, rels_indexed_batch)
            break
        break


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    print(FLAGS)
    main(FLAGS)
    print('Done')
