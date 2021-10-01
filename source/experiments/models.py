import random

import torch.nn as nn
import torch
import numpy as np
from sklearn import metrics


class LSTM(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, n_layers, dropout, batch_first=False):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.name = "LSTM"

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=batch_first)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, lens):

        packed_input = nn.utils.rnn.pack_padded_sequence(X, lens.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_input)
        output, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return self.sigmoid(self.fc_out(output))


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, n_layers, dropout, batch_first=False):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.name = "MLP"

        layers = [nn.Linear(input_dim, self.hid_dim), nn.Dropout(dropout), nn.ReLU()]

        for layer in range(max(0, n_layers - 1)):
            layers += [nn.Linear(self.hid_dim, self.hid_dim), nn.Dropout(dropout), nn.ReLU()]

        layers += [nn.Linear(self.hid_dim, output_dim), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, X, lens):

        output = self.net(X)

        packed_output = nn.utils.rnn.pack_padded_sequence(output, lens.to('cpu'), batch_first=True, enforce_sorted=False)
        output, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output


class GLM(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, n_layers, dropout, batch_first=False):
        super().__init__()

        self.name = "GLM"

        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, X, lens):

        output = self.net(X)

        packed_output = nn.utils.rnn.pack_padded_sequence(output, lens.to('cpu'), batch_first=True, enforce_sorted=False)
        output, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output


class RNN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, n_layers, dropout, batch_first=False):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.name = "RNN"

        self.rnn = nn.RNN(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=batch_first)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, lens):

        packed_input = nn.utils.rnn.pack_padded_sequence(X, lens.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)
        output, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        #return self.sigmoid(self.fc_out(hidden[-1]))
        return self.sigmoid(self.fc_out(output))


def train(model, loader, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0

    for i, (X, y, lens) in enumerate(loader):
        optimizer.zero_grad()

        y_pred = model(X, lens)
        loss = 0
        for ii, ll in enumerate(lens):
            seq_true, seq_pred = y[ii,:int(ll)].squeeze(), y_pred[ii,:int(ll)].squeeze()
            loss += criterion(seq_pred, seq_true) / len(lens)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):

    model.eval()

    epoch_loss, accuracy, precision, recall, f1_score = 0, 0, 0, 0, 0
    n_batches = len(loader)
    with torch.no_grad():

        for i, (X, y, lens) in enumerate(loader):

            y_pred = model(X, lens)
            batch_size = len(lens)

            for ii, ll in enumerate(lens):
                seq_true, seq_pred = y[ii,:int(ll)].squeeze(), y_pred[ii,:int(ll)].squeeze()

                # add loss to epoch loss
                epoch_loss += criterion(seq_pred, seq_true).item() / batch_size

                # make binary predictions
                seq_pred = torch.round(seq_pred)

                # calculate metrics
                accuracy += metrics.accuracy_score(seq_true, seq_pred) / batch_size
                precision += metrics.precision_score(seq_true, seq_pred) / batch_size
                recall += metrics.recall_score(seq_true, seq_pred) / batch_size
                f1_score += metrics.f1_score(seq_true, seq_pred) / batch_size

    # calculate average loss and metrics
    epoch_loss /= n_batches
    accuracy /= n_batches
    precision /= n_batches
    recall /= n_batches
    f1_score /= n_batches

    return epoch_loss, [accuracy, precision, recall, f1_score]
