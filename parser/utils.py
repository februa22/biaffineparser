# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn.init import orthogonal_

import _pickle as pickle
import json
import csv

# for debugging
import pdb
import os

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
            vec = np.full(self.max_length,
                          self.vocab[self.pad_token], dtype=np.int32)
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
                vec = np.concatenate((vec[:np.where(vec == pad)[0][0]], [
                                     end], vec[np.where(vec == pad)[0][0]:-1]))
        return vec

    def transform(self, X):
        return np.asarray([self.__look_up(x) for x in X], dtype=np.int32)


def load_dataset(filepath, flags):

    # check dataset_multiindex and save or load
    dataset_multiindex_filepath = 'embeddings/dataset_multiindex.pkl'
    if os.path.isfile(dataset_multiindex_filepath):
        print('dataset_multiindex_file exists, loading dataset_multiindex_file...')
        (sentences, pos, rels, heads, maxlen, maxwordlen,
         sentences_only) = load_vocab(dataset_multiindex_filepath)
        # pass
    else:
        print('Creating dataset_multiindex_file...')
        (sentences, pos, rels, heads, maxlen, maxwordlen,
         sentences_only) = get_dataset_multiindex(filepath)
        print(
            f'Saving dataset_multiindex_file... : {dataset_multiindex_filepath}')
        save_vocab((sentences, pos, rels, heads, maxlen, maxwordlen,
                    sentences_only), dataset_multiindex_filepath)

    # pdb.set_trace()
    #sentences_only = [ [word for word in sentences]]

    _, heads_features_dict, _ = initialize_embed_features(
        heads, 100, maxlen, starti=0, return_embeddings=False)

    _, pos_features_dict, pos_embedding_matrix = initialize_embed_features(
        pos, flags.pos_embed_size, maxlen, split_word=True)

    pos_indexed = get_indexed_sequences(
        pos, vocab=pos_features_dict, maxl=maxlen, maxwordl=maxwordlen, split_word=True)

    rels_indexed, rels_features_dict, _ = initialize_embed_features(
        rels, 100, maxlen, starti=0)

    heads_padded = get_indexed_sequences(
        heads, vocab=heads_features_dict, maxl=maxlen, just_pad=True)

    _, words_dict, words_embeddings_matrix = initialize_embed_features(
        sentences, flags.word_embed_size, maxlen, split_word=True, starti=0)

    # word_dic for sentences only
    _, words_only_dict, _ = initialize_embed_features(
        sentences_only, flags.word_embed_size, maxlen, split_word=True, starti=0)

    # get word_embeddings from pretrained glove file and add glove vocabs to word_dict
    # words only embedding
    flags.word_only_embed_file = 'embeddings/words.morph.original.vec'
    if flags.word_only_embed_file:
        words_only_embedding_matrix, words_only_dict = load_embed_model(
            flags.word_only_embed_file, words_dict=words_only_dict, embedding_size=200)
    if flags.word_embed_file:
        words_embeddings_matrix, words_dict = load_embed_model(
            flags.word_embed_file, words_dict=words_dict, embedding_size=flags.word_embed_size)
    if flags.pos_embed_file:
        pos_embedding_matrix, pos_features_dict = load_embed_model(
            flags.pos_embed_file, words_dict=pos_features_dict, embedding_size=flags.pos_embed_size)

    # making word_dictionary
    sentences_indexed = get_indexed_sequences(
        sentences, words_dict, maxl=maxlen, maxwordl=maxwordlen, split_word=True)

    sentences_only_indexed = get_indexed_sequences(
        sentences_only, words_only_dict, maxl=maxlen, maxwordlen=maxwordlen, split_word=True)

    # saving words and pos embeddings matrix
    flags.word_embed_matrix_file = 'embeddings/word_only_embed_matrix'
    print(f'saving words_only_embeddings_matrix: {word_embed_matrix_file}')
    np.savetxt(word_embed_matrix_file, words_only_embedding_matrix)

    if flags.word_embed_matrix_file:
        print(
            f'saving words_embeddings_matrix: {flags.word_embed_matrix_file}')
        np.savetxt(flags.word_embed_matrix_file, words_embeddings_matrix)
    if flags.pos_embed_matrix_file:
        print(f'saving pos_embeddings_matrix: {flags.pos_embed_matrix_file}')
        np.savetxt(flags.pos_embed_matrix_file, pos_embedding_matrix)

    # dumping dictionaries
    print('dumping words_dict')
    # words_only_dict
    with open('embeddings/words_only_dict.json', 'w') as f:
        json.dump(words_only_dict, f, indent=4, ensure_ascii=False)
    with open('embeddings/words_dict.json', 'w') as f:
        json.dump(words_dict, f, indent=4, ensure_ascii=False)
    print('dumping pos_features_dict')
    with open('embeddings/pos_features_dict.json', 'w') as f:
        json.dump(pos_features_dict, f, indent=4)
    print('dumping heads_features_dict')
    with open('embeddings/heads_features_dict.json', 'w') as f:
        json.dump(heads_features_dict, f, indent=4)
    print('dumping rels_features_dict')
    with open('embeddings/rels_features_dict.json', 'w') as f:
        json.dump(rels_features_dict, f, indent=4)
    return sentences_indexed, pos_indexed, heads_padded, rels_indexed, words_dict, pos_features_dict, heads_features_dict, rels_features_dict, words_embeddings_matrix, pos_embedding_matrix, maxlen, words_only_dict, sentences_only_indexed
    pdb.set_trace()

# loading embed model


def load_embed_model(embed_file_path, words_dict, embedding_size):
    print("Loading Pre-trained Embedding Model and merging with current dict")
    glove_dict = {}
    with open(embed_file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            if line.strip():
                word_and_embedding = line.strip().split('\t', 1)
                word = word_and_embedding[0]
                embedding = np.array(
                    [a for a in word_and_embedding[1].split(',')])
                glove_dict[word] = embedding
                # add glove word in word_dict
                if word not in words_dict:
                    words_dict[word] = len(words_dict)
        # get embedding_size: ex) 200
        #embedding_size = len(embedding)

    # create empty embedding matrix with zeros
    # create random embedding matrix for initialization
    np.random.seed(0)  # for reproducibility
    embedding_matrix = np.random.rand(len(words_dict), embedding_size)
    #embedding_matrix = np.zeros((len(words_dict), embedding_size))
    for key, value in words_dict.items():
        word_vocab = key
        word_index = value
        word_vector = glove_dict.get(word_vocab, None)
        # add word_vector to matrix
        if word_vector is not None:
            embedding_matrix[word_index] = word_vector
    #unk_index = words_dict['<UNK>']
    #embedding_matrix[unk_index] = np.random.rand(embedding_size)
    # replace padding in embedding matrix into np.zeros
    pad_index = words_dict['<PAD>']
    embedding_matrix[pad_index] = np.zeros(embedding_size)
    return embedding_matrix, words_dict


def save_vocab(vocab, filepath):
    print_out(f'Save vocab... {filepath}')
    # with open(filepath, 'wb') as f:
    #    pickle.dump(vocab, f)
    pickle.dump(vocab, open(filepath, 'wb'))


def load_vocab(filepath):
    print_out(f'Load vocab... {filepath}')
    return pickle.load(open(filepath, 'rb'))


def get_indexed_sequences(sequences: list, vocab: dict, maxl: int, just_pad=False, split_word=False, maxwordl=0):
    """
    Index and pad sequences according to vocab and max len
    :param sequences:
    :param vocab:
    :param maxl:
    :param just_pad:
    :param split_word: split word into morphs: shape(?,?,?)
    :param maxwordl: max length of splitted words
    :return:
    """
    # 어절 내에서도 split할 경우
    if split_word:
        indexed_sequences = np.full((len(sequences), maxl, maxwordl), vocab.get(
            '<PAD>', GLOBAL_PAD_SYMBOL), dtype=np.int32)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(sequence):
                # print(sequence)
                for k, v in enumerate(str(s).strip().split('|')):
                    if k >= maxwordl:
                        break
                    if just_pad:
                        indexed_sequences[i, j, k] = v
                    else:
                        indexed_sequences[i, j, k] = vocab.get(
                            v, vocab.get('<UNK>', GLOBAL_UNK_SYMBOL))
    else:
        indexed_sequences = np.full((len(sequences), maxl), vocab.get(
            '<PAD>', GLOBAL_PAD_SYMBOL), dtype=np.int32)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(sequence):
                if j >= maxl:
                    break
                if just_pad:
                    indexed_sequences[i, j] = s
                else:
                    indexed_sequences[i, j] = vocab.get(
                        s, vocab.get('<UNK>', GLOBAL_UNK_SYMBOL))
    return indexed_sequences


def initialize_embed_features(features: list, dim: int, maxl: int, starti: int=0, return_embeddings: bool=True, split_word: bool=False):
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
            # 어절을 잘라야할 경우
            if split_word:
                for word in str(f).strip().split('|'):
                    if features_dict.get(word, None) is None:
                        features_dict[word] = i
                        i += 1
            else:
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
    dataset = pd.read_csv(filepath, sep='\t', quoting=csv.QUOTE_NONE)
    # Process to make eoj string
    dataset['eoj'] = dataset['eoj'].apply(lambda x: str(x))
    # Process to make head_id float
    dataset['head_id'] = dataset['head_id'].apply(lambda x: int(x))
    dataset = dataset.set_index(['sent_id'])
    sentences = []
    sentences_only = []
    pos = []
    rels = []
    heads = []
    maxlen = 0
    maxwordlen = 0
    for i in dataset.index.unique():
        temp_sent = ['ROOT_START'] + cast_safe_list(dataset.loc[i]['eoj'])
        temp_sent_only = ['ROOT_START'] + \
            cast_safe_list(dataset.loc[i]['eoj_only'])
        temp_pos = ['ROOT_START'] + cast_safe_list(dataset.loc[i]['pos'])
        temp_rels = ['ROOT_START'] + cast_safe_list(dataset.loc[i]['label'])
        temp_heads = [0] + cast_safe_list(dataset.loc[i]['head_id'])
        sentences.append(temp_sent)
        pos.append(temp_pos)
        rels.append(temp_rels)
        heads.append(temp_heads)
        sentences_only.append(temp_sent_only)
        tempsentlen = len(temp_sent)
        # get longest size of the word (어절)
        tempwordlen = max([len(word.strip().split('|')) for word in temp_sent])
        if tempsentlen > maxlen:
            maxlen = tempsentlen
        if tempwordlen > maxwordlen:
            maxwordlen = tempwordlen
        if i % 5000 == 0:
            print("reading index=", i)
    # maxwordlen added for getting word length(어절 내의 최대 단어 수)
    return sentences, pos, rels, heads, maxlen, maxwordlen, sentences_only


def replace_and_save_dataset(input_file, heads, rels, output_file):
    print_out(f'Replace dataset... {input_file}')
    # dataset = pd.read_csv(input_file)
    dataset = pd.read_csv(input_file, sep='\t', quoting=csv.QUOTE_NONE)
    dataset = dataset.set_index(['sent_id'])
    for i in dataset.index.unique():
        dataset.loc[i, 'label'] = rels[i]
        dataset.loc[i, 'head_id'] = heads[i]

    print_out(f'Save dataset... {output_file}')
    dataset.to_csv(output_file, encoding='utf-8')


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def init_lstm_weights(lstm, initializer=orthogonal_):
    for layer_p in lstm._all_weights:
        for p in layer_p:
            if 'weight' in p:
                initializer(lstm.__getattr__(p))


def get_sequence_length(data, pad_id, axis=1):
    sequence_length = np.sum(np.not_equal(data[:, :, 0], pad_id), axis=axis)
    # print(f"sequence_length={sequence_length}")
    return sequence_length


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
