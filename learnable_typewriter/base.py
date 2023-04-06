"""Stage 2"""
import datetime
import torch
from os.path import join, exists
from omegaconf import OmegaConf, open_dict
from learnable_typewriter.utils.defaults import RUNS_PATH
from learnable_typewriter.utils.logger import get_logger, pretty_print
from learnable_typewriter.utils.file import mkdir

def merge(k, kp):
    if len(k):
        return k + '_' + kp
    else:
        return kp

def get_aliases(dictionary, k=''):
    data = []
    for kp, v in dictionary.items():
        if 'path' in v:
            data.append(merge(k, kp))
        else:
            data += get_aliases(v, k=merge(k, kp))
    return data

class Base(object):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""
    def __init__(self, cfg):
        self.__load_cfg__(cfg)
        self.__init_device__()

    def make_timestamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def run_dir_(self):
        if 'run_dir' in self.cfg:
            return self.cfg['run_dir']
        elif 'tag' in self.cfg:
            if self.eval:
                timestamp = self.cfg['timestamp']
            else:
                timestamp = self.make_timestamp()

            path = join(self.cfg.get('default_run_dir', RUNS_PATH), self.dataset_alias, self.cfg["tag"], timestamp)

            if not self.eval:
                with open_dict(self.cfg):
                    self.cfg.timestamp = timestamp
                    self.cfg.run_dir = path

            return path

        raise Exception('Either tag or run_dir should be added to the configuration')

    @property
    def post_load_flag(self):
        return self.eval or (self.cfg["training"].get('pretrained') is not None)

    def __load_cfg__(self, cfg):
        self.cfg = cfg
        self.log_cfg = cfg['training']['log']
        self.eval = cfg.get('eval', False)
        self.n_workers = cfg["training"].get("n_workers", 4)
        self.epoch = None

        self.dataset_alias = '+'.join(get_aliases(self.cfg["dataset"], k=''))
        run_dir = self.run_dir_()
        
        if self.eval:
            if not exists(run_dir):
                run_dir_prev = run_dir
                run_dir = run_dir_prev.replace(self.cfg.get('default_run_dir', str(RUNS_PATH)), str(RUNS_PATH))
            assert exists(run_dir), f'Default {run_dir_prev}, {run_dir} doesn\'t point to anywhere.'

        self.run_dir = mkdir(run_dir)
        if not self.eval:
            self.logger = get_logger(self.run_dir, name="train")
        self.log(f"run directory set to {self.run_dir}")

    def __getattr__(self, __name):
        if ('model' in self.__dict__) and (__name in self.model.__dict__):
            return self.model.__name
        else:
            if 'model' in self.__dict__:
                raise AttributeError(f'Either \'{type(self).__name__}\' or \'{type(self.model).__name__}\' have an attribute named \'{__name}\'')
            else:
                raise AttributeError(f'Either \'{type(self).__name__}\' have an attribute named \'{__name}\'')

    def cache_config(self):
        if not self.eval:
            self.config_path = join(self.run_dir, 'config.yaml')
            with open(self.config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(self.cfg))

    def __init_device__(self):
        device_id = self.cfg['training'].get('device')
        self.device = torch.device((f"cuda:{device_id}" if (device_id not in [None, 'cpu']) else 'cpu'))
        self.log(f"device: {self.device}")

    def log(self, message, eval=None):
        if eval is None:
            # Allows for selecting certain logs for both eval and train / or for train only.
            eval = not self.eval

        if eval:
            if self.epoch is not None and not self.train_end:
                message = f"[{self.epoch}/{self.n_epochs} | {self.batch}/{self.n_batches}] " + message

            if self.eval:
                pretty_print(message)
            else:
                self.logger.info(message)
                

    def print_memory_usage(self, prefix):
        attributes = ["memory_allocated", "max_memory_allocated", "memory_cached", "max_memory_cached"]
        usage = {attr: getattr(torch.cuda, attr)() * 0.000001 for attr in attributes}
        self.log(f"{prefix}:\t" + " / ".join([f"{k}: {v:.0f}MiB" for k, v in usage.items()]))
