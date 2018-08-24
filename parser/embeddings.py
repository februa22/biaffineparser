# -*- coding: utf-8 -*-

import numpy as np
import torch
import tensorflow as tf
from torch import nn

'''
class EmbeddingsLayer(nn.Module):
    def __init__(self, words_vocab: dict, words_embeddings_matrix: np.array, pad_idx: int):
        """
        Non trainable Embeddings layer
        :param words_vocab:
        :param words_embeddings_matrix:
        :param pad_idx:
        """
        super(EmbeddingsLayer, self).__init__()
        self.EMBEDDING_SIZE = len(words_embeddings_matrix[0])
        self.words_dict = words_vocab
        self.word_embed = nn.Embedding(
            num_embeddings=len(self.words_dict),
            embedding_dim=self.EMBEDDING_SIZE,
            padding_idx=pad_idx,
        )
        self.word_embed.weight = nn.Parameter(torch.from_numpy(words_embeddings_matrix).type(torch.FloatTensor), requires_grad=False)
'''

#EmbeddingsLayer with tensorflow
class EmbeddingsLayer():
    """
    Non trainable Embeddings layer
    :param words_vocab:
    :param words_embeddings_matrix:
    :param pad_idx: => not used in tensorflow
    """
    def __init__(self, words_vocab: dict, words_embeddings_matrix: np.array, pad_idx: int):
        self.EMBEDDING_SIZE = len(words_embeddings_matrix[0])
        self.words_dict = words_vocab
        self.word_embeddings = tf.get_variable("word_embeddings", [len(self.words_dict), self.EMBEDDING_SIZE])
        self.embedded_word_ids = tf.nn.embedding_lookup(self.word_embeddings, list(self.words_dict.values()))
        self.word_embeddings_weight = tf.Variable(words_embeddings_matrix, trainable=False)

if __name__ == '__main__':
    exit()