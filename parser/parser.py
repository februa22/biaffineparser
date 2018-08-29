# -*- coding: utf-8 -*-
""" TensorFlow Deep Biaffine Attention model implementation. """
import argparse

import numpy as np

from .model import Model
from . import utils

import pdb

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
    parser.add_argument('--lstm_hidden_size', type=int, default=450,
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
    parser.add_argument("--num_train_epochs", type=int, default=400,
                        help="Num epochs to train.")

    # data
    parser.add_argument('--train_filename', type=str,
                        help='Path of train dataset')
    parser.add_argument('--dev_filename', type=str,
                        help='Path of dev dataset')

    # Default settings works well (rarely need to change)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")

    # Misc
    parser.add_argument('--device', type=str,
                        help='Device to use')

    # Debug
    parser.add_argument('--debug', type=str2bool,
                        help='Use debugger to track down bad values during training', default=True)

# for parsing bool values
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(flags):
    print('Loading dataset..')
    (sentences_indexed,
        pos_indexed,
        heads_padded,
        rels_indexed,
        words_dict,
        pos_features_dict,
        heads_features_dict,
        rels_features_dict,
        word_embedding,
        pos_embedding,
        maxlen,
     ) = utils.load_dataset(flags.train_filename)

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

    best_val_loss = 100
    stopcount = 0

    model = Model(flags, words_dict, pos_features_dict, rels_features_dict, heads_features_dict,
                  word_embedding, pos_embedding)
    model.train(sentences_indexed, pos_indexed, rels_indexed, heads_padded)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    print(FLAGS)
    main(FLAGS)
    print('Done')
