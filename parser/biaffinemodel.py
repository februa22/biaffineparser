# -*- coding: utf-8 -*-
import numpy as np

from parser.utils import init_lstm_weights

import torch
from torch import nn
from torch.nn.init import orthogonal_, xavier_uniform_


def biaffine(input1, W, input2, device, num_outputs=1, bias_x=False, bias_y=False):

    batch_size, batch_len, dim = input1.size()
    if bias_x:
        input1 = torch.cat((input1, torch.ones(batch_size, batch_len, 1).to(device)), 2)
    if bias_y:
        input2 = torch.cat((input2, torch.ones(batch_size, batch_len, 1).to(device)), 2)

    nx, ny = dim + bias_x, dim + bias_y

    W = W.contiguous().view(nx, num_outputs * W.size()[-1])
    lin = torch.matmul(input1.contiguous().view(batch_size * batch_len, nx), W)
    lin = lin.contiguous().view(batch_size, num_outputs * batch_len, ny)
    blin = torch.matmul(lin, torch.transpose(input2, 1, 2))
    blin = blin.contiguous().view(batch_size, batch_len, num_outputs, batch_len)
    if num_outputs == 1:
        blin = blin.squeeze(2)
    else:
        blin = blin.transpose(2, 3)

    return blin


class BiaffineParser(nn.Module):
    def __init__(self, pos_vocab, rels_vocab, heads_vocab, hyperparams, device='cpu'):
        super(BiaffineParser, self).__init__()

        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                print('WARNING: Cuda not installed, pytorch gpu not installed or', device,
                      'is not available. Setting cpu as device')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.LSTM_HIDDEN_SIZE = hyperparams.get('LSTM_HIDDEN_SIZE', 450)
        self.LSTM_DROPOUT = hyperparams.get('LSTM_DROPOUT', 0.55)
        self.DROPOUT = hyperparams.get('DROPOUT', 0.44)
        self.EMBEDDING_DROPOUT = hyperparams.get('EMBEDDING_DROPOUT', 0.44)
        self.EMBEDDING_SIZE = hyperparams.get('EMBEDDING_SIZE', 100)
        self.POS_EMBEDDING_SIZE = hyperparams.get('POS_EMBEDDING_SIZE', 100)
        self.ARC_MLP_UNITS = hyperparams.get('ARC_MLP_UNITS', 500)
        self.LABEL_MLP_UNITS = hyperparams.get('LABEL_MLP_UNITS', 100)
        self.N_CLASSES = len(rels_vocab)
        self.NUM_LAYERS = hyperparams.get('NUM_LAYERS', 3)
        self.rels_vocab = rels_vocab
        self.heads_vocab = heads_vocab

        self.pos_vocab = pos_vocab
        self.pos_embed = nn.Embedding(
            num_embeddings=len(pos_vocab),
            embedding_dim=self.POS_EMBEDDING_SIZE,
            padding_idx=pos_vocab['<PAD>'],
        )
        self.dropout = nn.Dropout(p=self.DROPOUT)
        self.embedding_dropout = nn.Dropout2d(p=self.EMBEDDING_DROPOUT)

        self.lstm = torch.nn.LSTM(
            input_size=self.EMBEDDING_SIZE + self.POS_EMBEDDING_SIZE,
            hidden_size=self.LSTM_HIDDEN_SIZE, num_layers=self.NUM_LAYERS, dropout=self.LSTM_DROPOUT, batch_first=True, bidirectional=True).to(
            self.device,
        )

        init_lstm_weights(self.lstm)

        self.arc_head_mlp = torch.nn.Linear(self.LSTM_HIDDEN_SIZE * 2, self.ARC_MLP_UNITS).to(self.device)
        self.arc_dep_mlp = torch.nn.Linear(self.LSTM_HIDDEN_SIZE * 2, self.ARC_MLP_UNITS).to(self.device)

        self.label_head_mlp = torch.nn.Linear(self.LSTM_HIDDEN_SIZE * 2, self.LABEL_MLP_UNITS).to(self.device)
        self.label_dep_mlp = torch.nn.Linear(self.LSTM_HIDDEN_SIZE * 2, self.LABEL_MLP_UNITS).to(self.device)

        self.loss_heads = torch.nn.CrossEntropyLoss(ignore_index=heads_vocab['<PAD>'])
        self.loss_rels = torch.nn.CrossEntropyLoss(ignore_index=rels_vocab['<PAD>'])

        self.initial_trainable_hidden_states = (torch.autograd.Variable(xavier_uniform_(torch.zeros(self.NUM_LAYERS * 2, 1, self.LSTM_HIDDEN_SIZE).to(self.device))),
                                                torch.autograd.Variable(xavier_uniform_(torch.zeros(self.NUM_LAYERS * 2, 1, self.LSTM_HIDDEN_SIZE).to(self.device))))

        self.W_arc = nn.Parameter(orthogonal_(
            torch.empty(self.ARC_MLP_UNITS + 1, 1, self.ARC_MLP_UNITS).to(self.device)))

        self.W_label = nn.Parameter(orthogonal_(
            torch.empty(self.LABEL_MLP_UNITS + 1, self.N_CLASSES, self.LABEL_MLP_UNITS + 1).to(self.device)))

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), betas=(0.9, 0.9),
                                          lr=0.002, weight_decay=0.000001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True,
                                                                    factor=0.9)

        self.activation = torch.nn.LeakyReLU()

    def init_hidden(self, batch_size: int):
        return (
            torch.nn.init.normal_(torch.zeros(self.NUM_LAYERS * 2, batch_size, self.LSTM_HIDDEN_SIZE).to(self.device), 0, 0.1),
            torch.nn.init.normal_(torch.zeros(self.NUM_LAYERS * 2, batch_size, self.LSTM_HIDDEN_SIZE).to(self.device), 0, 0.1),
        )

    def init_hidden_trainable(self, batch_size: int):

        return (
            torch.cat([self.initial_trainable_hidden_states[0]] * batch_size, dim=1),
            torch.cat([self.initial_trainable_hidden_states[1]] * batch_size, dim=1),
        )

    def forward(self, embedded_sents: torch.autograd.Variable, pos: list, gold_heads: list, data_lengths: np.ndarray):
        """
        Returns the logits for the labels and the arcs
        :return:
        """

        # Sort lens (pos lens are the same)
        sorted_indices = np.argsort(-np.array(data_lengths)).astype(np.int64)

        # Extract sorted lens by index
        lengths = data_lengths[sorted_indices]
        max_len = lengths[0]

        embedded_sents = self.embedding_dropout(embedded_sents[:, :max_len])

        pos_t = torch.LongTensor(pos)[:, :max_len]
        embedded_pos = self.pos_embed(torch.autograd.Variable(pos_t))
        embedded_pos = self.embedding_dropout(embedded_pos)

        # Extract tensors ordered by len
        stacked_x = embedded_sents.index_select(dim=0,
                                                index=torch.autograd.Variable(torch.from_numpy(sorted_indices))).to(
            self.device)
        stacked_pos_x = embedded_pos.index_select(dim=0,
                                                  index=torch.autograd.Variable(torch.from_numpy(sorted_indices))).to(
            self.device)

        # Apply dropout and when one is dropped scale the other
        mask_words = stacked_x - stacked_pos_x == stacked_x

        mask_pos = stacked_pos_x - stacked_x == stacked_pos_x

        stacked_x[mask_words] *= 2
        stacked_pos_x[mask_pos] *= 2

        stacked_x = torch.cat((stacked_x, stacked_pos_x), dim=2)

        stacked_x = nn.utils.rnn.pack_padded_sequence(stacked_x, torch.from_numpy(lengths).to(self.device),
                                                      batch_first=True)

        x_lstm, _ = self.lstm(stacked_x, self.init_hidden_trainable(len(embedded_sents)))

        x_lstm, _ = nn.utils.rnn.pad_packed_sequence(x_lstm, batch_first=True)

        # Reorder the batch
        x_lstm = x_lstm.index_select(dim=0, index=torch.autograd.Variable(
            torch.from_numpy(np.argsort(sorted_indices).astype(np.int64)).to(self.device)))

        # NN scoring
        h_arc_dep = self.arc_dep_mlp(x_lstm)
        h_arc_dep = self.activation(h_arc_dep)
        h_arc_dep = self.dropout(h_arc_dep)

        h_arc_head = self.arc_head_mlp(x_lstm)
        h_arc_head = self.activation(h_arc_head)
        h_arc_head = self.dropout(h_arc_head)

        h_label_dep = self.label_dep_mlp(x_lstm)
        h_label_dep = self.activation(h_label_dep)
        h_label_dep = self.dropout(h_label_dep)

        h_label_head = self.label_head_mlp(x_lstm)
        h_label_head = self.activation(h_label_head)
        h_label_head = self.dropout(h_label_head)

        # Heads computation
        s_i_arc = biaffine(h_arc_dep, self.W_arc, h_arc_head, self.device, num_outputs=1, bias_x=True)

        # Labels computation
        full_label_logits = biaffine(h_label_dep, self.W_label, h_label_head, self.device, num_outputs=self.N_CLASSES,
                                     bias_x=True, bias_y=True)

        if self.training:
            gold_heads_t = torch.LongTensor(gold_heads)[:, :max_len].to(self.device)
            m = (gold_heads_t == self.heads_vocab['<PAD>'])
            gold_heads_t[m] *= 0
            pred_arcs = gold_heads_t
        else:
            pred_arcs = s_i_arc.argmax(-1)

        # Gather label logits from predicted or gold heads
        pred_arcs = pred_arcs.unsqueeze(2).unsqueeze(3)  # [batch, sent_len, 1, 1]
        pred_arcs = pred_arcs.expand(-1, -1, -1, full_label_logits.size(-1))  # [batch, sent_len, 1, n_labels]
        selected_label_logits = torch.gather(full_label_logits, 2, pred_arcs).squeeze(2)  # [batch, n_labels, sent_len]

        return s_i_arc, selected_label_logits, max_len, data_lengths
