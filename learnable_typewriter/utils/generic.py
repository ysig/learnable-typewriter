import torch
from functools import wraps
from itertools import chain, zip_longest

from numpy.random import seed as np_seed
from numpy.random import get_state as np_get_state
from numpy.random import set_state as np_set_state
from random import seed as rand_seed
from random import getstate as rand_get_state
from random import setstate as rand_set_state
from torch import manual_seed as torch_seed
from torch import get_rng_state as torch_get_state
from torch import set_rng_state as torch_set_state
from omegaconf import DictConfig

class nonce(object):
    def __getattr__(self, _):
        return self.nop
    def nop(*args, **kw):
        pass


class use_seed:
    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int) and seed >= 0
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.rand_state = rand_get_state()
            self.np_state = np_get_state()
            self.torch_state = torch_get_state()
            self.torch_cudnn_deterministic = torch.backends.cudnn.deterministic
            rand_seed(self.seed)
            np_seed(self.seed)
            torch_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            rand_set_state(self.rand_state)
            np_set_state(self.np_state)
            torch_set_state(self.torch_state)
            torch.backends.cudnn.deterministic = self.torch_cudnn_deterministic

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop('seed', None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper


def alternate(*args):
    for v in chain.from_iterable(zip_longest(*args)):
        if v is not None:
            yield v


def cfg_flatten(cfg):
    if isinstance(cfg, (list, tuple)):
        return ','.join([str(l) for l in cfg])
    elif isinstance(cfg, (dict, DictConfig)):
        data = {}
        for k, v in cfg.items():
            out = cfg_flatten(v)
            if isinstance(out, dict):
                for ki, vi in out.items():
                    data[k + '.' + ki] = vi
            else:
                data[k] = out
        return data
    elif cfg is None:
        return 'None'
    else:
        return str(cfg)