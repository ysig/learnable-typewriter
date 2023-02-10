import random
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from functools import partial

def pad_right(x, max_w, pad_value=None):
    if pad_value is None:
        padding_mode = 'edge'
        fill = 0
    else:
        padding_mode = 'constant'
        fill = pad_value

    w = x.size()[-1]
    if max_w > w:
        c = x.size()[0]
        if c > 1 and isinstance(fill, (list, tuple)) and padding_mode == 'constant':
            x = torch.cat([pad(x[i].unsqueeze(0), (0, max_w - w), value=fill[i], mode=padding_mode) for i in range(c)], dim=0)
        else:
            x = pad(x, (0, max_w - w), value=fill, mode=padding_mode)

    if len(x.size()) == 3:
        x = x.unsqueeze(0)
    if len(x.size()) == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    return x

def collate_fn(inp, supervised=None, alias=None):
    xs, ys = [], []
    for x, y in inp:
        x = to_tensor(x)
        ys.append(y)
        xs.append(x.unsqueeze(0))
    return {'x': torch.cat(xs, dim=0), 'y': ys, 'supervised': supervised, 'cropped': True, 'alias': alias}

def collate_fn_pad_to_max(batch, supervised=None, alias=None, pad_value=None, max_w=0):
    xs, ys, ws = [], [], []
    for x, y in batch:
        x = to_tensor(x) 
        ys.append(y)
        W = x.size()[-1]
        ws.append(W)
        max_w = max(W, max_w)
        xs.append(x)
    xs = torch.cat([pad_right(x, max_w, pad_value) for x in xs], dim=0)
    return {'x': xs, 'y': ys, 'w': ws, 'supervised': supervised, 'cropped': False, 'alias': alias}

def pad_right_batch(batch, max_w):
    xs = []
    for i in range(len(batch)):
        x = batch[i]
        xs.append(pad_right(x, max_w))
    return torch.cat(xs, 0)

def get_k(iterator, k):
    output, flag = [], False
    for _ in range(k):
        try: 
            output.append(next(iterator))
        except StopIteration:
            flag = True
            break
    return output, flag


#ERR_FN = 'SQADADL-LTW.error'
#ERR_LOCK = FileLock('./dataloader_worker.lock')

def worker(args):
    i, dataset = args
    return dataset[i]

#   except KeyboardInterrupt:
#        pass
#    except:
#        print('[ERROR] exception caught in data loader worker, logged in:', ERR_FN)
#        with ERR_LOCK:
#            with open(ERR_FN, 'a') as f:
#                f.write('-'*50 + '\n\n' + traceback.format_exc() + '\n')

import multiprocessing as mp
from itertools import chain
from torchvision.transforms.functional import to_tensor
import copy
import math

def custom_error_callback(error):
    raise error

class SequentialAdaptiveDataLoader(object):
    def __init__(self, dataset, batch_size, supervised, num_workers, pad_value, alias, min_width=0, mean_width=1024, drop_last=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size_submit = 2*batch_size
        self.nominal_length = batch_size*(dataset.mean_width if mean_width is None else mean_width) 
        self.min_width = min_width
        self.pad_value = pad_value
        self.alias = alias
        
        self.drop_last = drop_last
        self.supervised = supervised

        self.num_workers = min(num_workers, mp.cpu_count())
        self.sampler = list(range(len(dataset)))
        self.dataset = dataset
        if shuffle:
            random.shuffle(self.sampler)
            self.map_fname = 'imap_unordered'
        else:
            self.map_fname = 'imap'

        self.drop_last = drop_last

    def make_out(self,):
        return {'x': torch.cat([pad_right(x, self.cache['max_w'], pad_value=self.pad_value) for x in self.cache['xs']], dim=0),
                'y': self.cache['ys'],
                'w': self.cache['ws'],
                'supervised': self.supervised,
                'alias': self.alias,
                'cropped': False}

    def __len__(self):
        return int(math.ceil(len(self.dataset)*1.0/self.batch_size))

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.terminate()
            self.pool.join()

    def set_sample(self, samples):
        idx, stop = get_k(samples, self.batch_size_submit - self.num_submitted)
        if len(idx):
            args = [(i, copy.deepcopy(self.dataset)) for i in idx]
            self.results = getattr(self.pool, self.map_fname)(worker, args)
            if self.iter is None:
                self.iter = iter(self.results)
            else:
                self.iter = chain(self.iter, iter(self.results))

        self.num_submitted += len(idx)
        return stop

    def init_cache(self):
        self.num_submitted = 0
        self.iter = None
        self.pool = mp.Pool(self.num_workers)
        self.cache = {'xs': [], 'ys': [], 'ws': [], 'max_w': self.min_width}

    def get_sample(self, final=False):
        while self.num_submitted > 0:
            self.num_submitted -= 1 
            x, y = next(self.iter)
            x = to_tensor(x) # !!! If done inside the dataloader the whole thing freezes :/
            w = x.size()[-1]

            new_max_w = max(w, self.cache['max_w'])
            if new_max_w * (len(self.cache['xs']) + 1) > self.nominal_length:
                assert len(self.cache['xs'])
                output = self.make_out()
                self.cache.update({'xs': [x], 'ys': [y], 'ws': [w], 'max_w': max(w, self.min_width)})
                return output 

            self.cache['xs'].append(x)
            self.cache['ys'].append(y)
            self.cache['ws'].append(w)
            self.cache['max_w'] = new_max_w

        if final:
            return self.make_out()

    def __iter__(self):
        self.init_cache() 
        samples = iter(self.sampler)
        stop = self.set_sample(samples)
        while not stop:
            output = self.get_sample()
            data = self.get_sample()
            if data is not None:
                yield data
            stop = self.set_sample(samples)

        if self.drop_last and self.num_submitted > 0:
            yield self.get_sample(final=True)

        self.__del__()
