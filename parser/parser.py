# -*- coding: utf-8 -*-
""" TensorFlow Deep Biaffine Attention model implementation. """
import argparse
import os
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
    parser.add_argument('--embedding_dropout', type=float, default=.33,
                        help='Dropout rate for embedding')
    parser.add_argument('--lstm_dropout', type=float, default=.33,
                        help='Dropout rate for LSTM')
    parser.add_argument('--mlp_dropout', type=float, default=.33,
                        help='Dropout rate for MLP')
    parser.add_argument('--num_lstm_layers', type=int, default=3,
                        help='Number of LSTM layers')

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="sgd | adam")
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
    parser.add_argument("--word_vocab_name", type=str, default='word.pkl',
                        help="Vocab name of words.")
    parser.add_argument("--pos_vocab_name", type=str, default='pos.pkl',
                        help="Vocab name of pos.")
    parser.add_argument("--rel_vocab_name", type=str, default='rel.pkl',
                        help="Vocab name of rels.")
    parser.add_argument("--head_vocab_name", type=str, default='head.pkl',
                        help="Vocab name of heads.")

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


def evaluate(model, data, batch_size):
    (val_sentences_indexed, val_pos_indexed,
     val_rels_indexed, val_heads_padded) = data
    total_eval_loss, total_eval_uas, total_eval_las = [], [], []
    total_head_preds, total_rel_preds = [], []
    lengths = []
    # iterate over the dev-set
    for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
            val_sentences_indexed, val_pos_indexed, val_rels_indexed, val_heads_padded,
            batch_size=batch_size):
        batch_data = (sentences_indexed_batch, pos_indexed_batch,
                      heads_indexed_batch, rels_indexed_batch)
        result = model.eval_step(batch_data)

        (eval_loss, eval_uas, eval_las, head_preds,
         rel_preds, batch_lengths) = result
        total_eval_loss.append(eval_loss)
        total_eval_uas.append(eval_uas)
        total_eval_las.append(eval_las)
        total_head_preds.extend(head_preds)
        total_rel_preds.extend(rel_preds)
        lengths.extend(batch_lengths)
    eval_loss = np.mean(total_eval_loss)
    eval_uas = np.mean(total_eval_uas)
    eval_las = np.mean(total_eval_las)
    return eval_loss, eval_uas, eval_las, total_head_preds, total_rel_preds, lengths


def evaluate_and_write_predictions(flags, model, data, inference_input_file):
    model.new_sess_and_restore(flags.out_dir)
    eval_result = evaluate(model, data, flags.batch_size)
    (_, uas, las, all_head_preds, all_rel_preds_ids, lengths) = eval_result
    utils.print_out(f'# uas: {uas} - las: {las}')

    rev_vocab_rels = {i: w for w, i in (model.rels_vocab_table.items())}
    all_rel_preds = [rev_vocab_rels[i] for i in all_rel_preds_ids]

    utils.replace_and_save_dataset(inference_input_file,
                                   all_head_preds, all_rel_preds,
                                   os.path.join(flags.out_dir, 'dev_prediction.csv'))


def main(flags):
    # loading trainind dataset and embed
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

    # embed vadliation(dev) dataset
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

    dev_data = (val_sentences_indexed, val_pos_indexed,
                val_rels_indexed, val_heads_padded)

    best_eval_uas = .0
    stop_count = 0

    model = Model(flags, words_dict, pos_features_dict,
                  rels_features_dict, heads_features_dict, word_embedding)
    model.build()

    utils.save_vocab(words_dict, os.path.join(
        flags.out_dir, flags.word_vocab_name))
    utils.save_vocab(pos_features_dict, os.path.join(
        flags.out_dir, flags.pos_vocab_name))
    utils.save_vocab(rels_features_dict, os.path.join(
        flags.out_dir, flags.rel_vocab_name))
    utils.save_vocab(heads_features_dict, os.path.join(
        flags.out_dir, flags.head_vocab_name))

    # train
    for epoch in range(flags.num_train_epochs):
        # reset progbar each epoch
        progbar = Progbar(len(sentences_indexed))
        sentences_indexed, pos_indexed, rels_indexed, heads_padded = shuffle(
            sentences_indexed, pos_indexed, rels_indexed, heads_padded, random_state=0)

        # iterate over the train-set
        for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
                sentences_indexed, pos_indexed, rels_indexed, heads_padded, batch_size=flags.batch_size):
            batch_data = (sentences_indexed_batch, pos_indexed_batch,
                          heads_indexed_batch, rels_indexed_batch)
            _, loss, uas, las, global_step = model.train_step(batch_data)

            if global_step % 10 == 0:
                model.add_summary(global_step, 'train/loss', loss)
                model.add_summary(global_step, 'train/uas', uas)
                model.add_summary(global_step, 'train/las', las)
            progbar.add(len(sentences_indexed_batch), values=[
                ('epoch', int(epoch)),
                ('loss', loss),
                ('uas', uas),
                ('las', las),
            ])
        # eval_step
        eval_result = evaluate(model, dev_data, flags.batch_size)
        (eval_loss, eval_uas, eval_las, _, _, _) = eval_result
        model.add_summary(global_step, 'eval/loss', eval_loss)
        model.add_summary(global_step, 'eval/uas', eval_uas)
        model.add_summary(global_step, 'eval/las', eval_las)
        progbar.add(1, values=[
            ('epoch', int(epoch)),
            ('loss', loss),
            ('uas', uas),
            ('las', las),
            ('eval_loss', eval_loss),
            ('eval_uas', eval_uas),
            ('eval_las', eval_las),
        ])
        print('\n')
        # save the best model or early stopping
        if eval_uas > best_eval_uas:
            model.save()
            best_eval_uas = eval_uas
            stop_count = 0
            utils.print_out('# new best UAS!\n')
        elif stop_count >= 20:
            utils.print_out(f'# early stopping {stop_count} \
                            epochs without improvement\n')
            break
        else:
            stop_count += 1
    evaluate_and_write_predictions(flags, model, dev_data, flags.dev_filename)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    print('#'*30)
    print(FLAGS)
    print('#'*30)
    main(FLAGS)
    print('Done')
