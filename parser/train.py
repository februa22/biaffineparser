# -*- coding: utf-8 -*-

import os

import numpy as np

from parser.biaffinemodel import BiaffineParser
from parser.embeddings import EmbeddingsLayer
from parser.progress_bar import Progbar
from parser.utils import get_dataset_multiindex, get_indexed_sequences, load_dataset, get_batch

import _pickle as pickle

from sklearn.utils import shuffle

import torch


def loss_acc(model: torch.nn.Module, logits: torch.Tensor, gold_labels: np.array, acc: list, batch_lens: list, max_len: int, loss_type: str):
    """
    Function that returns the loss value and the accuracy of a minibatch for a pytorch model
    for a sequence tagging task
    :param model:
    :param logits:
    :param gold_labels:
    :param acc:
    :param loss_type:
    :param batch_lens:
    :param max_len:
    :return:
    """
    gold_batch_np = np.array(gold_labels)[:, :max_len].astype(np.int64)
    gold_batch_tensor = torch.autograd.Variable(
        torch.LongTensor(torch.from_numpy(gold_batch_np)).to(model.device))

    if loss_type == 'heads':
        mask = gold_batch_tensor[:, 1:] != model.heads_vocab['<PAD>']
        loss = model.loss_heads(logits.transpose(1, 2), gold_batch_tensor)
    elif loss_type == 'rels':
        mask = gold_batch_tensor[:, 1:] != model.rels_vocab['<PAD>']
        loss = model.loss_rels(logits.transpose(1, 2), gold_batch_tensor)
    else:
        raise RuntimeError('Pass a loss type with name heads or rels')

    preds = logits.argmax(-1).cpu().data.numpy()
    mask = mask.data.cpu().numpy()
    preds_equals = np.equal(preds[:, 1:] * mask, gold_batch_np[:, 1:] * mask)

    for l, r in zip(batch_lens, preds_equals):
        acc.append(sum(r[:(l - 1)]) / (l - 1))
    return loss, acc


if __name__ == '__main__':
    print('Loading dataset..')

    (sentences_indexed,
     pos_indexed,
     heads_padded,
     rels_indexed,
     words_dict,
     pos_features_dict,
     heads_features_dict,
     rels_features_dict,
     words_embeddings_matrix,
     pos_embedding_matrix,
     maxlen,
     ) = load_dataset('data/train_conll17.csv')

    val_sentences, val_pos, val_rels, val_heads, val_maxlen = get_dataset_multiindex(
        'data/dev_conll17.csv',
    )

    val_sentences_indexed = get_indexed_sequences(val_sentences, words_dict, val_maxlen)
    val_pos_indexed = get_indexed_sequences(val_pos, pos_features_dict, val_maxlen)
    val_rels_indexed = get_indexed_sequences(val_rels, rels_features_dict, val_maxlen)
    val_heads_padded = get_indexed_sequences(val_heads, heads_features_dict, val_maxlen, just_pad=True)

    test_sentences, test_pos, test_rels, test_heads, test_maxlen = get_dataset_multiindex(
        'data/test_conll17.csv',
    )

    test_sentences_indexed = get_indexed_sequences(test_sentences, words_dict, test_maxlen)
    test_pos_indexed = get_indexed_sequences(test_pos, pos_features_dict, test_maxlen)
    test_rels_indexed = get_indexed_sequences(test_rels, rels_features_dict, test_maxlen)
    test_heads_padded = get_indexed_sequences(test_heads, heads_features_dict, test_maxlen, just_pad=True)

    print('Done.')
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

    hyperparams = {'EMBEDDING_SIZE': 100,
                   'EMBEDDING_TRAIN_SIZE': 100,
                   'LSTM_HIDDEN_SIZE': 450,
                   'POS_EMBEDDING_SIZE': 100,
                   'ARC_MLP_UNITS': 500,
                   'LABEL_MLP_UNITS': 100,
                   'DROPOUT': .44,
                   'EMBEDDING_DROPOUT': .33,
                   'NUM_LSTM_LAYERS': 3}

    biaffinemodel = BiaffineParser(pos_features_dict, rels_features_dict, heads_features_dict, hyperparams,
                                   device='gpu')
    embeddings_layer = EmbeddingsLayer(words_dict, words_embeddings_matrix, words_dict['<PAD>'])
    embeddings_layer.eval()
    if not os.path.isdir("parser/tempmodels/"):
        os.mkdir("parser/tempmodels/")
    pickle.dump(hyperparams, open('parser/tempmodels/hyperparams', 'wb'))
    pickle.dump(heads_features_dict, open('parser/tempmodels/heads_features_dict', 'wb'))
    pickle.dump(rels_features_dict, open('parser/tempmodels/rels_features_dict', 'wb'))

    epochs = 400
    best_val_loss = 100
    stopcount = 0

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
        for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in get_batch(
                sentences_indexed, pos_indexed, rels_indexed, heads_padded, batch_size=100):
            heads_bar = []
            rels_bar = []
            biaffinemodel.train(True)
            biaffinemodel.optimizer.zero_grad()

            lengths = np.sum(np.not_equal(sentences_indexed_batch, embeddings_layer.words_dict['<PAD>']), axis=1)
            embedded_sents = embeddings_layer(sentences_indexed_batch)

            logits, labels_logits, max_len_batch, batch_lens = biaffinemodel.forward(embedded_sents,
                                                                                     pos_indexed_batch,
                                                                                     heads_indexed_batch,
                                                                                     lengths)

            loss_heads, heads_bar = loss_acc(biaffinemodel, logits, heads_indexed_batch, heads_bar, batch_lens,
                                             max_len_batch, 'heads')
            loss_rels, rels_bar = loss_acc(biaffinemodel, labels_logits, rels_indexed_batch, rels_bar, batch_lens,
                                           max_len_batch, 'rels')

            total_loss = loss_heads + loss_rels

            model_loss.append([loss_heads.data.cpu().numpy(), loss_rels.data.cpu().numpy()])

            total_loss.backward()
            biaffinemodel.optimizer.step()

            progbar.add(len(sentences_indexed_batch),
                        values=[
                            ('epoch', int(epoch)),
                            ('loss', total_loss.data.cpu().numpy()),
                            ('rels_acc', sum(rels_bar) / len(rels_bar)),
                            ('heads_acc', sum(heads_bar) / len(heads_bar)),
                        ])

        # Start val
        for sentences_indexed_batch, pos_indexed_batch, rels_indexed_batch, heads_indexed_batch in get_batch(
                val_sentences_indexed, val_pos_indexed, val_rels_indexed, val_heads_padded, batch_size=128):
            biaffinemodel.eval()

            embedded_sents = embeddings_layer(sentences_indexed_batch)
            lengths = np.sum(np.not_equal(sentences_indexed_batch, embeddings_layer.words_dict['<PAD>']), axis=1)

            logits, labels_logits, max_len_batch, batch_lens = biaffinemodel.forward(embedded_sents,
                                                                                     pos_indexed_batch,
                                                                                     heads_indexed_batch,
                                                                                     lengths)

            loss_heads, val_heads_acc = loss_acc(biaffinemodel, logits, heads_indexed_batch, val_heads_acc, batch_lens,
                                                 max_len_batch, 'heads')
            loss_rels, val_rels_acc = loss_acc(biaffinemodel, labels_logits, rels_indexed_batch, val_rels_acc,
                                               batch_lens, max_len_batch, 'rels')

            model_loss = loss_heads + loss_rels
            val_model_loss.append([loss_heads.data.cpu().numpy(), loss_rels.data.cpu().numpy()])

        total_val_loss = np.sum(val_model_loss) / len(val_model_loss)

        biaffinemodel.scheduler.step(total_val_loss)

        progbar.add(1, values=[
            ('epoch', int(epoch)),
            ('loss', total_loss.data.cpu().numpy()),
            ('heads_acc', sum(heads_acc)),
            ('rels_acc', sum(rels_acc)),
            ('val_loss', total_val_loss),
            ('val_rels_acc', sum(val_rels_acc) / len(val_rels_acc)),
            ('val_heads_acc', sum(val_heads_acc) / len(val_heads_acc)),
        ])
        print('\n')

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(biaffinemodel.state_dict(), 'parser/tempmodels/biaffineparser_' + str(epoch))
            stopcount = 0
        else:
            stopcount += 1
            if stopcount > 20:
                break
