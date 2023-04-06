import numpy as np
import torch
from learnable_typewriter.evaluate.quantitative.sprite_matching.loss import matching_loss, matching_loss_batch

class MatchingModelRobust(torch.nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.S, self.A = S, A
        self.P = torch.nn.Parameter(torch.softmax(torch.rand(S, A), dim=1))
        self.do_nothing_ij = torch.from_numpy(np.array([1, 1]))

    def forward(self, xs, ys, x_lengths, y_lengths):
        cij = 1 - torch.softmax(self.P, dim=1)
        dij = torch.clamp(self.do_nothing_ij, 0, 1)
        return matching_loss(xs, ys, x_lengths, y_lengths, cij, dij)

class MatchingModel(torch.nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.S, self.A = S, A
        self.P = torch.nn.Parameter(torch.softmax(torch.rand(S, A), dim=1))
        self.do_nothing_ij = torch.from_numpy(np.array([1, 1]))

    def forward(self, xs, ys, x_lengths, y_lengths):
        return matching_loss_batch(xs, ys, x_lengths-1, y_lengths-1, torch.nn.functional.pad(1 - torch.softmax(self.P, dim=1), (0, 1, 0, 1), value=float('Inf')), self.do_nothing_ij)
