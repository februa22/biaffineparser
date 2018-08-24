# -*- coding: utf-8 -*-

import numpy as np
import torch
import tensorflow as tf

#embedding layer with tensorflow
class EmbeddingsLayer():
    """
    Non trainable Embeddings layer
    :param words_vocab:
    :param words_embeddings_matrix:
    :param pad_idx:
    """
    def __init__(self, words_vocab: dict, words_embeddings_matrix: np.array, pad_idx: int):
        #super(EmbeddingsLayer, self).__init__()
        self.EMBEDDING_SIZE = len(words_embeddings_matrix[0])
        self.words_dict = words_vocab
        self.word_embed = nn.Embedding(
            num_embeddings=len(self.words_dict),
            embedding_dim=self.EMBEDDING_SIZE,
            padding_idx=pad_idx,
        )
        #self.word_embed.weight = nn.Parameter(torch.from_numpy(words_embeddings_matrix).type(torch.FloatTensor), requires_grad=False)
        self.word_embed.weight = tf.Variable(words_embeddings_matrix, trainable=False)

if __name__ == '__main__':
    exit()