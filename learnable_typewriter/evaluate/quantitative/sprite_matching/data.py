import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def make_dataloader(dataset, batch_size=2, collate_fn=None, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=min(num_workers, batch_size))

def make_alphabet(s):
    return {a: i for i, a in enumerate(sorted(list(s)))}, {i: a for i, a in enumerate(sorted(list(s)))}

class Data(object):
    def __init__(self, trainer, data_iterator, tag=''):
        self.data = []

        if tag is not None:
            data_iterator = tqdm(data_iterator, desc=f"build-dataset:{tag}")
        
        torch.cuda.empty_cache()
        for x in data_iterator:
            preds = trainer.inference(x)
            self.data += list(zip(preds, x['y']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, A, S):
        self.A, self.S = A, S
        self.data = data

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
