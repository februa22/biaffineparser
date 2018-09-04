# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn.init import orthogonal_

import _pickle as pickle
import csv

#for debugging
import pdb

GLOBAL_PAD_SYMBOL = '<PAD>'
GLOBAL_UNK_SYMBOL = '<UNK>'


def get_batch(*args, batch_size=1):

    for i in range(0, len(args[0]), batch_size):
        yield (dataset[i:i + batch_size] for dataset in args)


class ConllEntry:
    def __init__(self, w_id, form, pos, cpos, parent_id=None, relation=None):
        self.id = w_id
        self.form = form
        self.cpos = cpos.upper()
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation
        self.children = []
        self.pred_parent_id = -1
        self.pred_relation = 0

    def __str__(self):
        return str(self.__dict__)


class VocabSelector:
    def __init__(self, vocab, max_length, oov_token='<UNK>', pad_token=None, end_token=None, tokenized=True):
        self.vocab = vocab
        self.max_length = max_length
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.end_token = end_token
        self.tokenized = tokenized

    def __look_up(self, x):
        if self.pad_token:
            vec = np.full(self.max_length, self.vocab[self.pad_token], dtype=np.int32)
        else:
            vec = np.zeros(self.max_length, dtype=np.int32)
        if not self.tokenized:
            tokens = x.split()
        else:
            tokens = x
        for index, word in enumerate(tokens):
            if index >= self.max_length:
                break
            vec[index] = self.vocab.get(word, self.vocab[self.oov_token])
        if self.end_token:
            pad = self.vocab[self.pad_token]
            end = self.vocab[self.end_token]
            if pad in vec:
                vec = np.concatenate((vec[:np.where(vec == pad)[0][0]], [end], vec[np.where(vec == pad)[0][0]:-1]))
        return vec

    def transform(self, X):
        return np.asarray([self.__look_up(x) for x in X], dtype=np.int32)


def load_dataset(filepath):
    sentences, pos, rels, heads, maxlen = get_dataset_multiindex(filepath)

    pos_indexed, pos_features_dict, pos_embedding_matrix = initialize_embed_features(pos, 100, maxlen)
    #pdb.set_trace() #

    rels_indexed, rels_features_dict, _ = initialize_embed_features(rels, 100, maxlen, starti=0)
    #pdb.set_trace() #

    _, heads_features_dict, _ = initialize_embed_features(heads, 100, maxlen, starti=0)
    heads_padded = get_indexed_sequences(heads, vocab=heads_features_dict, maxl=maxlen, just_pad=True)

    words_dict = pickle.load(open('embeddings/vocab.pkl', 'rb'))

    sentences_indexed = get_indexed_sequences(sentences, words_dict, maxlen)
    words_embeddings_matrix = np.load('embeddings/vectors.npy', allow_pickle=False)
    return sentences_indexed, pos_indexed, heads_padded, rels_indexed, words_dict, pos_features_dict, heads_features_dict, rels_features_dict, words_embeddings_matrix, pos_embedding_matrix, maxlen


def save_vocab(vocab, filepath):
    print_out(f'Save vocab... {filepath}')
    pickle.dump(vocab, open(filepath, 'wb'))


def load_vocab(filepath):
    print_out(f'Load vocab... {filepath}')
    return pickle.load(open(filepath, 'rb'))


def get_indexed_sequences(sequences: list, vocab: dict, maxl: int, just_pad=False):
    """
    Index and pad sequences according to vocab and max len
    :param sequences:
    :param vocab:
    :param maxl:
    :param just_pad:
    :return:
    """
    indexed_sequences = np.full((len(sequences), maxl), vocab.get('<PAD>', GLOBAL_PAD_SYMBOL), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(sequence):
            if j >= maxl:
                break
            if just_pad:
                indexed_sequences[i, j] = s
            else:
                indexed_sequences[i, j] = vocab.get(s, vocab.get('<UNK>', GLOBAL_UNK_SYMBOL))

    return indexed_sequences


def initialize_embed_features(features: list, dim: int, maxl: int, starti: int=0, return_embeddings: bool=True):
    """
    Takes a list of sequences, for example sentences, pos tags or relations.
    Initialize a dict and the random embedding matrix to train
    :param features:
    :param dim: dimension of the initialized embeddings
    :param maxl: maximum length
    :param starti: index form where to start
    :return: Indexed features, vocab, embeddings
    """
    features_dict = {}
    i = starti
    for sentence in features:
        for f in sentence:
            if features_dict.get(f, None) is None:
                features_dict[f] = i
                i += 1
    features_dict['<UNK>'] = len(features_dict)
    features_dict['<PAD>'] = len(features_dict)
    indexed = get_indexed_sequences(features, features_dict, maxl)
    if return_embeddings:
        embedding_matrix = np.random.randn(len(features_dict), dim)
    else:
        embedding_matrix = None
    return indexed, features_dict, embedding_matrix


def cast_safe_list(elem):
    if type(elem) != pd.Series:
        elem = pd.Series(elem)
    return list(elem)


def get_dataset_multiindex(filepath):
    print_out(f'Load dataset... {filepath}')
    # dataset = pd.read_csv(filepath, sep=',')
    dataset = pd.read_csv(filepath, sep='\t', quoting=csv.QUOTE_NONE)
    # Only preprocess I make is lowercase
    dataset['w'] = dataset['w'].apply(lambda x: str(x).lower())
    dataset = dataset.set_index(['s'])
    sentences = []
    pos = []
    rels = []
    heads = []
    maxlen = 0
    for i in dataset.index.unique():
        temp_sent = ['ROOT_START'] + cast_safe_list(dataset.loc[i]['w'])
        temp_pos = ['ROOT_START'] + cast_safe_list(dataset.loc[i]['x'])
        temp_rels = ['ROOT_START'] + cast_safe_list(dataset.loc[i]['f'])
        temp_heads = [0] + cast_safe_list(dataset.loc[i]['g'])
        sentences.append(temp_sent)
        pos.append(temp_pos)
        rels.append(temp_rels)
        heads.append(temp_heads)
        tempsentlen = len(temp_sent)
        if tempsentlen > maxlen:
            maxlen = tempsentlen
    return sentences, pos, rels, heads, maxlen


def replace_and_save_dataset(input_file, heads, rels, output_file):
    print_out(f'Replace dataset... {input_file}')
    dataset = pd.read_csv(input_file)
    dataset = dataset.set_index(['s'])
    for i in dataset.index.unique():
        dataset.loc[i, 'f'] = rels[i]
        dataset.loc[i, 'g'] = heads[i]
    
    print_out(f'Save dataset... {output_file}')
    dataset.to_csv(output_file, encoding='utf-8')


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def init_lstm_weights(lstm, initializer=orthogonal_):
    for layer_p in lstm._all_weights:
        for p in layer_p:
            if 'weight' in p:
                initializer(lstm.__getattr__(p))


def get_sequence_length(data, pad_id, axis=1):
    return np.sum(np.not_equal(data, pad_id), axis=axis)


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()
