import copy
import random
from os.path import join, isdir
from functools import partial
from omegaconf.listconfig import ListConfig

import numpy as np
import torch
from torch.utils.data import DataLoader

from learnable_typewriter.base import Base
from learnable_typewriter.data.dataset import LineDataset, ExemplarDataset
from learnable_typewriter.data.dataloader import collate_fn, collate_fn_pad_to_max
from learnable_typewriter.utils.defaults import DATASETS_PATH
from learnable_typewriter.data.dataloader import SequentialAdaptiveDataLoader

def merge(k, kp):
    if len(k):
        return k + '_' + kp
    else:
        return kp

def flatten_dataset(dictionary, k=''):
    data = []
    for kp, v in dictionary.items():
        if 'path' in v:
            data.append((merge(k, kp), dict(v)))
        else:
            data += flatten_dataset(v, k=merge(k, kp))
    return data

def split_2(x):
    a, b = zip(*x)
    return list(a), list(b)


class Dataset(Base):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transcribe = None
        self.__set_pad_value__()
        self.__init_dataset_params__()
        if not self.post_load_flag:
            self.__init_dataloaders__()

    def __post_init_dataset__(self):
        self.__init_dataloaders__()

    @property
    def supervised(self):
        return not self.unsupervised

    def __set_pad_value__(self):
        freeze = self.cfg['model']['background'].get('init', {'freeze': False}).get('freeze', False)
        init_val = self.cfg['model']['background'].get('init', {'constant': None}).get('constant', None)
        self.pad_value = (init_val if (freeze and (init_val is not None)) else None)
        if isinstance(self.pad_value, ListConfig):
            self.pad_value = tuple(self.pad_value)

    def __init_dataset_params__(self):
        # Datasets and dataloaders
        self.alias, self.dataset_kwargs = split_2(flatten_dataset(self.cfg["dataset"]))
        self.default_dataset_path = self.cfg.get('default_dataset_path', DATASETS_PATH)
        if self.eval:
            if not isdir(self.default_dataset_path):
                old = self.default_dataset_path
                self.default_dataset_path = DATASETS_PATH
                assert isdir(self.default_dataset_path), f'Default {old}, {DATASETS_PATH} doesn\'t point to anywhere.'

        for i in range(len(self.dataset_kwargs)):
            self.dataset_kwargs[i]['path'] = join(self.default_dataset_path, self.dataset_kwargs[i]['path'])
            self.dataset_kwargs[i]['supervised'] = self.dataset_kwargs[i].get('supervised', False)
            self.dataset_kwargs[i]['alias'] = self.alias[i]
            if 'padding' in self.dataset_kwargs[i] and isinstance(self.dataset_kwargs[i]['padding'], ListConfig):
                self.dataset_kwargs[i]['padding'] = tuple(self.dataset_kwargs[i]['padding'])

        self.height = self.cfg['model']['encoder']['H']
        self.min_width = self.cfg['model']['transformation']['canvas_size'][1]

        assert sum(data['supervised'] for data in self.dataset_kwargs) <= 1
        self.unsupervised_dataset = all(not data['supervised'] for data in self.dataset_kwargs)
        self.unsupervised = self.unsupervised_dataset
        self.dataset_kwargs = sorted(self.dataset_kwargs, key=lambda data: int(data['supervised']), reverse=True)

    def __init_dataloaders__(self):
        self.batch_size = self.cfg["training"]["batch_size"]
        self.experimental = self.cfg["training"]["adaptive_dataloader"]
        self.canvas_size = self.cfg['model']['transformation']['canvas_size']
        self.train_loader = self.get_dataloader(split='train', batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
        self.val_loader = self.get_dataloader(split='val', batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False, remove_crop=True)
        self.test_loader = self.get_dataloader(split='test', batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False, remove_crop=True)

        self.log(f"initializing dataloaders with batch_size={self.batch_size} and num_workers={self.n_workers}")
        
        if self.supervised:
            self.log('Number of sprites : {}'.format(len(self.transcribe)))

        self.val_flag = (sum(len(v.dataset) for v in self.val_loader) > 0 or sum(len(v.dataset) for v in self.test_loader) > 0) and self.cfg["training"].get("evaluate", {}).get("active", True)
        self.has_labels = [getattr(self.train_loader[i].dataset, 'has_labels', True) for i in range(len(self.train_loader))]
        if all(self.has_labels) != any(self.has_labels):
            raise NotImplementedError('Partially unlabeled datasets are not yet correctly implemented')
        self.has_labels = all(self.has_labels)

        self.__get_set_dataset_size__()

    def __get_set_dataset_size__(self):
        dataset_size = []
        for dataloader in self.train_loader:
            if isinstance(dataloader.dataset, LineDataset):
                dataset_size.append(len(dataloader.dataset))

        if len(dataset_size):
            num_samples = int(np.mean(dataset_size))
            for dataloader in self.train_loader:
                if isinstance(dataloader.dataset, ExemplarDataset):
                    dataloader.dataset.num_samples = num_samples

    @property
    def img_size(self):
        self.img_size = self.train_loader.dataset.img_size

    def get_dataset(self, dataset_args, split, dataset_size=None):
        exemplar = dataset_args.pop('exemplar', False)
        has_transcribe = 'transcribe_dataset' in self.__dict__
        transcribe = (self.transcribe_dataset if has_transcribe else None)
        if exemplar:
            dataset = ExemplarDataset(split=split, height=self.height, transcribe=transcribe, **dataset_args)
        else:
            dataset = LineDataset(split=split, height=self.height, dataset_size=dataset_size, transcribe=transcribe, padding_value=self.pad_value, **dataset_args)

        if not has_transcribe:
            assert split == 'train'
            self.transcribe_dataset = dataset.transcribe
            if self.supervised:
                self.transcribe = self.transcribe_dataset
            
        return dataset

    def get_dataloader(self, split, percentage=1.0, random_subset=False, batch_size=None, shuffle=False, num_workers=None, dataset_size=None, remove_crop=False):
        dataloaders = []
        for data in self.dataset_kwargs:
            data = dict(data)
            if data.pop('split', split) == split:
                supervised = data.get('supervised', False)
                dataloaders.append(self.get_dataloader_(data, split, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, dataset_size=dataset_size, supervised=supervised, percentage=percentage, random_subset=random_subset, remove_crop=remove_crop))
        return dataloaders

    def get_dataloader_(self, dataset_args, split, percentage=1.0, random_subset=False, batch_size=None, shuffle=False, num_workers=None, dataset_size=None, supervised=False, remove_crop=False): 
        assert 0 < percentage <= 1.0

        if num_workers is None:
            num_workers = self.n_workers

        if batch_size is None:
            batch_size = self.batch_size

        if remove_crop:
            dataset_args = copy.deepcopy(dataset_args)
            dataset_args.pop('crop_width', None)
        dataset = self.get_dataset(dataset_args, split, dataset_size)
        cropped = dataset.cropped

        #extracts the subset
        if percentage < 1:
            indices = list(range(len(dataset)))
            if random_subset:
                random.shuffle(indices)

            indices = indices[:int(len(indices)*percentage)]
            dataset = torch.utils.data.Subset(dataset, indices)

        if not 'train_loader' in self.__dict__ or not 'val_loader' in self.__dict__ or not 'test_loader' in self.__dict__:
            self.log('{} set loaded, len dataset : {}. Instances of width <= {} and N_min >= {}. Images padded to width {} minimum.'.format(
                split, len(dataset), dataset_args.get('W_max',float('inf')), dataset_args.get('N_min',0), self.min_width))

        num_workers = min(num_workers, batch_size) if num_workers != 0 else num_workers
        if cropped:
            assert split == 'train'
            return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=partial(collate_fn, supervised=supervised, alias=dataset_args['alias']))
        else:
            if split == 'train' and self.experimental:
                return SequentialAdaptiveDataLoader(dataset, batch_size, supervised, num_workers, min_width=self.min_width, pad_value=self.pad_value, alias=dataset_args['alias'])
            else:
                return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=partial(collate_fn_pad_to_max, supervised=supervised, pad_value=self.pad_value, alias=dataset_args['alias'], max_w=self.canvas_size[1]))
