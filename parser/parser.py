# -*- coding: utf-8 -*-
""" TensorFlow Deep Biaffine Attention model implementation. """
import argparse
import json
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
    parser.add_argument('--num_lstm_units', type=int, default=400,
                        help='Number of hidden units for LSTM')
    parser.add_argument('--num_lstm_layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--arc_mlp_units', type=int, default=500,
                        help='Number of hidden units for MLP of arc')
    parser.add_argument('--label_mlp_units', type=int, default=100,
                        help='Number of hidden units for MLP of label')
    parser.add_argument('--word_embed_size', type=int, default=100,
                        help="The embedding dimension for the word's embedding.")
    #new
    parser.add_argument('--char_embed_size', type=int, default=100,
                        help="The embedding dimension for the char's embedding.")                    
    parser.add_argument('--pos_embed_size', type=int, default=100,
                        help="The embedding dimension for the POS's embedding.")

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
    #new
    parser.add_argument("--char_vocab_name", type=str, default='char.pkl',
                        help="Vocab name of chars.")
    parser.add_argument("--pos_vocab_name", type=str, default='pos.pkl',
                        help="Vocab name of pos.")
    parser.add_argument("--rel_vocab_name", type=str, default='rel.pkl',
                        help="Vocab name of rels.")
    parser.add_argument("--head_vocab_name", type=str, default='head.pkl',
                        help="Vocab name of heads.")
    parser.add_argument("--word_embed_file", type=str, default=None,
                        help="Use the pre-trained embedding. \
                        If not provided, use random values.")
    #new
    parser.add_argument("--char_embed_file", type=str, default=None,
                        help="Use the pre-trained embedding. \
                        If not provided, use random values.")
    parser.add_argument("--pos_embed_file", type=str, default=None,
                        help="Use the pre-trained embedding. \
                        If not provided, use random values.")
    parser.add_argument("--word_embed_matrix_file", type=str, default=None,
                        help="word_embed_martix file path (numpy to text). \
                        If not provided, not saving.")
    #new
    parser.add_argument("--char_embed_matrix_file", type=str, default=None,
                        help="char_embed_martix file path (numpy to text). \
                        If not provided, not saving.")                         
    parser.add_argument("--pos_embed_matrix_file", type=str, default=None,
                        help="pos_embed_martix file path (numpy to text). \
                        If not provided, not saving.")

    # Default settings works well (rarely need to change)
    parser.add_argument('--embed_dropout', type=float, default=.33,
                        help='Dropout rate for the embedding (not keep_prob)')
    parser.add_argument('--lstm_dropout', type=float, default=.33,
                        help='Dropout rate for LSTM (not keep_prob)')
    parser.add_argument('--mlp_dropout', type=float, default=.33,
                        help='Dropout rate for MLP (not keep_prob)')
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")

    # Misc
    parser.add_argument('--device', type=str,
                        help='Device to use')
    parser.add_argument('--debug', type=str2bool,
                        help='Use debugger to track down bad values during training', default=True)

    # Inference
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")


def str2bool(v):
    """ for parsing bool values """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate(model, data, batch_size):
    (val_sentences_indexed, val_chars_indexed, val_pos_indexed,
     val_rels_indexed, val_heads_padded) = data
    total_eval_loss, total_eval_uas, total_eval_las = [], [], []
    total_head_preds, total_rel_preds = [], []
    lengths = []
    # iterate over the dev-set
    for sentences_indexed_batch, chars_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
            val_sentences_indexed, val_chars_indexed, val_pos_indexed, val_rels_indexed, val_heads_padded,
            batch_size=batch_size):
        batch_data = (sentences_indexed_batch, chars_indexed_batch,
                      pos_indexed_batch, heads_indexed_batch, rels_indexed_batch)
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


def evaluate_and_write_predictions(flags, model, data, inference_input_file, inference_output_file):
    model.new_sess_and_restore(os.path.join(flags.out_dir, 'parser.ckpt'))
    eval_result = evaluate(model, data, flags.batch_size)
    (_, uas, las, all_head_preds, all_rel_preds_ids, lengths) = eval_result
    utils.print_out(f'# uas: {uas} - las: {las}')

    rev_vocab_rels = {i: w for w, i in (model.rels_vocab_table.items())}
    all_rel_preds = [rev_vocab_rels[i] for i in all_rel_preds_ids]

    utils.replace_and_save_dataset(inference_input_file,
                                   all_head_preds, all_rel_preds,
                                   inference_output_file)


def train(flags, log_f=None):
    log_filepath = os.path.join(flags.out_dir, 'log.txt')
    log_f = open(log_filepath, 'wb')
    utils.print_out('TRAIN started...', log_f)
    utils.print_out('#'*30, log_f)
    utils.print_out(str(flags), log_f)
    utils.print_out('#'*30, log_f)

    # loading trainind dataset and embed
    (sentences_indexed,  # (12543, 160)
        chars_indexed,
        pos_indexed,  # (12543, 160)
        heads_padded,   # (12543, 160)
        rels_indexed,   # (12543, 160)
        words_dict,  # 400005
        chars_dict,
        pos_features_dict,  # 20
        heads_features_dict,  # 132
        rels_features_dict,  # 53
        word_embedding,  # (400005, 100)
        chars_embedding,
        pos_embedding,  # (20, 100)
        maxlen  # 160
     ) = utils.load_dataset(flags.train_filename, flags)

    utils.print_out('#'*30, log_f)
    utils.print_out(f'sentences_indexed {sentences_indexed.shape}', log_f)
    utils.print_out(
        f'chars_indexed {chars_indexed.shape}', log_f)
    utils.print_out(f'pos_indexed {pos_indexed.shape}', log_f)
    utils.print_out(f'rels_indexed {rels_indexed.shape}', log_f)
    utils.print_out(f'heads_padded {heads_padded.shape}', log_f)
    utils.print_out('#'*30, log_f)

    # sanity check
    for i, h in enumerate(heads_padded):
        if len(h) != maxlen:
            utils.print_out(h, log_f)
            utils.print_out('ERROR IN PADDING HEADS', log_f)
        if len(rels_indexed[i]) != maxlen:
            utils.print_out(rels_indexed[i], log_f)
            utils.print_out('ERROR IN PADDING RELATIONS', log_f)
        if len(np.not_equal(rels_indexed[i], rels_features_dict['<PAD>'])) != len(
                np.not_equal(h, heads_features_dict['<PAD>'])):
            utils.print_out('ERROR IN LEN OF', log_f)
            utils.print_out(f'rels {rels_indexed[i]}', log_f)
            utils.print_out(f'heads {h}', log_f)

    # embed vadliation(dev) dataset
    val_sentences, val_chars, val_pos, val_rels, val_heads, val_maxlen, val_maxwordlen, val_maxcharlen = utils.get_dataset_multiindex(
        flags.dev_filename)

    val_sentences_indexed = utils.get_indexed_sequences(
        val_sentences, words_dict, val_maxlen, maxwordl=val_maxwordlen, split_word=True)
    val_chars_indexed = utils.get_indexed_sequences(
        val_chars, chars_dict, val_maxlen, maxwordl=val_maxcharlen, split_word=True)
    val_pos_indexed = utils.get_indexed_sequences(
        val_pos, pos_features_dict, val_maxlen, maxwordl=val_maxwordlen, split_word=True)
    val_rels_indexed = utils.get_indexed_sequences(
        val_rels, rels_features_dict, val_maxlen)
    val_heads_padded = utils.get_indexed_sequences(
        val_heads, heads_features_dict, val_maxlen, just_pad=True)
    #pdb.set_trace()
    
    dev_data = (val_sentences_indexed, val_chars_indexed, val_pos_indexed, val_rels_indexed, val_heads_padded)


    best_eval_uas = .0
    stop_count = 0

    model = Model(
        flags,
        words_dict,
        chars_dict,
        pos_features_dict,
        rels_features_dict,
        heads_features_dict,
        word_embedding,
        chars_embedding,
        pos_embedding)
    model.build()

    utils.save_vocab(words_dict, os.path.join(
        flags.out_dir, flags.word_vocab_name))
    utils.save_vocab(chars_dict, os.path.join(
        flags.out_dir, flags.char_vocab_name))
    utils.save_vocab(pos_features_dict, os.path.join(
        flags.out_dir, flags.pos_vocab_name))
    utils.save_vocab(rels_features_dict, os.path.join(
        flags.out_dir, flags.rel_vocab_name))
    utils.save_vocab(heads_features_dict, os.path.join(
        flags.out_dir, flags.head_vocab_name))
    # save hparams as json in out_dir
    hparams_json_path = os.path.join(flags.out_dir, 'hparams.json')
    print(f'Save hparams... {hparams_json_path}')
    json.dump(vars(flags), open(hparams_json_path, 'w',
                                encoding='utf-8'), indent=4, ensure_ascii=False)

    # train
    for epoch in range(flags.num_train_epochs):
        epoch += 1
        # reset progbar each epoch
        progbar = Progbar(len(sentences_indexed))
        sentences_indexed, chars_indexed, pos_indexed, rels_indexed, heads_padded = shuffle(
            sentences_indexed, chars_indexed, pos_indexed, rels_indexed, heads_padded, random_state=0)

        # iterate over the train-set
        for sentences_indexed_batch, chars_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in utils.get_batch(
                sentences_indexed, chars_indexed, pos_indexed, rels_indexed, heads_padded, batch_size=flags.batch_size):
            batch_data = (sentences_indexed_batch, chars_indexed_batch, pos_indexed_batch,
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
            s = f'epoch: {int(epoch)} - loss: {loss} - uas: {uas} - las: {las} - global_step: {global_step}'
            log_f.write(s.encode("utf-8"))
            log_f.write(b"\n")
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
        utils.print_out(
            f'epoch: {int(epoch)} - loss: {loss} - uas: {uas} - las: {las} - eval_loss: {eval_loss} - eval_uas: {eval_uas} - eval_las: {eval_las}',
            log_f)
        # save the best model or early stopping
        if eval_uas > best_eval_uas:
            model.save(os.path.join(flags.out_dir, 'parser.ckpt'))
            best_eval_uas = eval_uas
            stop_count = 0
            utils.print_out('# new best UAS!', log_f)
        elif stop_count >= 20:
            utils.print_out(
                f'# early stopping {stop_count} epochs without improvement',
                log_f)
            break
        else:
            stop_count += 1
    log_f.close()
    dev_output_file = os.path.join(flags.out_dir, 'dev.inference.tsv')
    evaluate_and_write_predictions(flags, model, dev_data, flags.dev_filename, dev_output_file)


def inference(flags, log_f=None):
    utils.print_out('INFERENCE started...')

    words_dict = utils.load_vocab(os.path.join(
        flags.out_dir, flags.word_vocab_name))
    pos_features_dict = utils.load_vocab(os.path.join(
        flags.out_dir, flags.pos_vocab_name))
    char_features_dict = utils.load_vocab(os.path.join(
        flags.out_dir, flags.char_vocab_name))
    rels_features_dict = utils.load_vocab(os.path.join(
        flags.out_dir, flags.rel_vocab_name))
    heads_features_dict = utils.load_vocab(os.path.join(
        flags.out_dir, flags.head_vocab_name))

    word_embedding, _ = utils.load_embed_model(flags.word_embed_file, words_dict, flags.word_embed_size)
    pos_embedding, _ = utils.load_embed_model(flags.pos_embed_file, pos_features_dict, flags.pos_embed_size)
    char_embedding, _ = utils.load_embed_model(flags.char_embed_file, char_features_dict, flags.char_embed_size)

    val_sentences, val_chars, val_pos, val_rels, val_heads, val_maxlen, val_maxwordlen, val_maxcharlen = utils.get_dataset_multiindex(
        flags.inference_input_file)

    val_sentences_indexed = utils.get_indexed_sequences(
        val_sentences, words_dict, val_maxlen, maxwordl=val_maxwordlen, split_word=True)
    val_char_indexed = utils.get_indexed_sequences(
        val_chars, char_features_dict, val_maxlen, maxwordl=val_maxcharlen, split_word=True)
    val_pos_indexed = utils.get_indexed_sequences(
        val_pos, pos_features_dict, val_maxlen, maxwordl=val_maxwordlen, split_word=True)
    val_rels_indexed = utils.get_indexed_sequences(
        val_rels, rels_features_dict, val_maxlen)
    val_heads_padded = utils.get_indexed_sequences(
        val_heads, heads_features_dict, val_maxlen, just_pad=True)

    test_data = (val_sentences_indexed, val_char_indexed, val_pos_indexed, val_rels_indexed, val_heads_padded)

    model = Model(
        flags,
        words_dict,
        char_features_dict,
        pos_features_dict,
        rels_features_dict,
        heads_features_dict,
        word_embedding,
        char_embedding,
        pos_embedding)

    evaluate_and_write_predictions(flags, model, test_data, flags.inference_input_file, flags.inference_output_file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    FLAGS = argparser.parse_args()
    if FLAGS.inference_input_file:
        inference(FLAGS)
    else:
        train(FLAGS)
    utils.print_out('Done')
