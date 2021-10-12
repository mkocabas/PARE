from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet as TorchDirichlet


class SmoothOneHot(nn.Module):
    def __init__(self, label_smoothing, random_off_targets=False):
        super(SmoothOneHot, self).__init__()
        self.register_buffer("_one_hot", torch.FloatTensor())
        self.register_buffer("_ones", torch.FloatTensor())
        self.register_buffer("_noise", torch.FloatTensor())
        self._label_smoothing = label_smoothing
        self._random_off_targets = random_off_targets

    def forward(self, indices, num_classes):
        if not self._random_off_targets:
            # construct one hot
            batch_size = indices.size(0)
            self._one_hot.resize_(batch_size, num_classes).fill_(0.0)
            self._ones.resize_(batch_size, num_classes).fill_(1.0)
            self._one_hot.scatter_(1, indices.view(-1,1), self._ones)
            one_hot_labels = self._one_hot
            # label smoothing
            smooth_positives = 1.0 - self._label_smoothing
            smooth_negatives = self._label_smoothing / num_classes
            return one_hot_labels * smooth_positives + smooth_negatives
        else:
            # construct one hot
            batch_size = indices.size(0)
            self._one_hot.resize_(batch_size, num_classes).fill_(0.0)
            self._ones.resize_(batch_size, num_classes).fill_(1.0)
            self._one_hot.scatter_(1, indices.view(-1,1), self._ones)
            positive_labels = self._one_hot
            smooth_positives = 1.0 - self._label_smoothing
            smooth_negatives = self._label_smoothing
            positive_labels = positive_labels * smooth_positives

            negative_labels = 1.0 - self._one_hot
            self._noise.resize_(batch_size, num_classes).uniform_(1e-1, 1.0)
            self._noise = self._noise * negative_labels
            self._noise = smooth_negatives * self._noise / self._noise.sum(dim=1, keepdim=True)
            one_hot_labels = positive_labels + self._noise
            return one_hot_labels

            # label smoothing
            # smooth_positives = 1.0 - self._label_smoothing
            # sum_negatives = self._label_smoothing

            # self._noise.resize_(batch_size, num_classes).random_(1e-5, 1-3)
            # self._noise = self._noise / self._noise.sum()
            # torch.random()

            # batch_size = indices.size(0)
            # self._one_hot.resize_(batch_size, num_classes).fill_(0.0)
            # self._ones.resize_(batch_size, num_classes).fill_(1.0)
            # self._one_hot.scatter_(1, indices.view(-1,1), self._ones)

            # torch.rand(1e-5, smooth_negatives)
            # offsets = torch.from_numpy(np.random.randint(low=1, high=num_classes, size=[batch_size])).cuda()
            # false_indices = torch.fmod((indices + offsets).float(), float(num_classes)).long()
            # self._ones.resize_(batch_size, num_classes).fill_(smooth_negatives)
            # self._one_hot.scatter_(1, false_indices.view(-1,1), self._ones)

            # one_hot_labels = self._one_hot * 1.0

            return one_hot_labels


class Dirichlet(nn.Module):
    def __init__(self, argmax_smoothing=1e-5, random_off_targets=False):
        super(Dirichlet, self).__init__()
        self._onehot = SmoothOneHot(
            label_smoothing=argmax_smoothing, random_off_targets=random_off_targets)
        self.register_buffer("_range", torch.FloatTensor())

    def forward(self, alpha, value):
        dirichlet = TorchDirichlet(alpha)
        return dirichlet.log_prob(value)

    def mean(self, alpha):
        return alpha / alpha.sum(-1, keepdim=True)
