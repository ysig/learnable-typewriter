from tqdm import tqdm

import torch

from learnable_typewriter.evaluate.quantitative.sprite_matching.data import make_dataloader
from learnable_typewriter.evaluate.quantitative.sprite_matching.model import MatchingModel, MatchingModelRobust

class TrainerBatch(object):
    def __init__(self, dataset, device=None, verbose=False):
        self.dataset = dataset
        self.verbose = verbose
        self.model = MatchingModel(self.dataset.S, self.dataset.A)
        self.device = device
        self.model = self.model.to(device)
        self.dataloader_kargs = {}

    def train(self, num_epochs=10, batch_size=256, lr=1, print_progress=False):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        dl = make_dataloader(self.dataset, batch_size=batch_size, collate_fn=self.collate_dataset, **self.dataloader_kargs)

        if print_progress:
            epochs = tqdm(range(1, num_epochs + 1), desc=f'Training matching model: lr={lr}')
        else:
            epochs = range(1, num_epochs + 1)

        for _ in epochs:
            for xs, ys, x_lengths, y_lengths in dl:
                xs, ys, x_lengths, y_lengths = xs.to(self.device), ys.to(self.device), x_lengths.to(self.device), y_lengths.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model(xs, ys, x_lengths, y_lengths)
                if loss.requires_grad:
                    loss.backward()
                    self.optimizer.step()

        self.mapping = self.export_mapping_()

    def collate_dataset(self, x):
        xs, ys, x_lengths, y_lengths = [], [], [], []
        Nmax = max(len(x[i][0]) for i in range(len(x)))
        Mmax = max(len(x[i][1]) for i in range(len(x)))

        for i in range(len(x)):
            x1, x2 = x[i]
            xs.append(torch.LongTensor(x1 + [self.model.S]*(Nmax - len(x1))).unsqueeze(-1))
            ys.append(torch.LongTensor(x2 + [self.model.A]*(Mmax - len(x2))).unsqueeze(-1))
            x_lengths.append(len(x1))
            y_lengths.append(len(x2))

        return torch.cat(xs, dim=-1), torch.cat(ys, dim=-1), torch.LongTensor(x_lengths), torch.LongTensor(y_lengths)

    def export_mapping_(self):
        return dict(enumerate(torch.argmax(torch.softmax(self.model.P, dim=1), dim=1).tolist()))


class Trainer(TrainerBatch):
    def __init__(self, dataset, device=None, verbose=False):
        self.dataset = dataset
        self.model = MatchingModelRobust(self.dataset.S, self.dataset.A)
        self.verbose = verbose
        self.device = 'cpu'
        self.dataloader_kargs = {'num_workers': 0}
