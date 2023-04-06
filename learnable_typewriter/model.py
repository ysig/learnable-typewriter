from pathlib import Path
from shutil import copyfile
from os.path import join
from time import time
from collections import defaultdict

import torch
from learnable_typewriter.utils.defaults import MODEL_FILE, BEST_MODEL
from learnable_typewriter.typewriter.inference import InferenceUnsupervised, InferenceSupervised
from learnable_typewriter.typewriter.model import LearnableTypewriter
from learnable_typewriter.typewriter.utils import count_parameters
from learnable_typewriter.dataset import Dataset


class Model(Dataset):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""
    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.post_load_flag:
            self.__post_init_model__()
        self.distribution_centers = defaultdict(list) 

    def __post_init_model__(self):
        self.__init_model__()
        self.__init_inference__()

    def __init_inference__(self):
        self.inference = (InferenceUnsupervised if self.unsupervised else InferenceSupervised)(self.model)

    def __init_model__(self):
        self.model = LearnableTypewriter(self.cfg['model'], self.transcribe, logger=self.log).to(self.device)
        self.log(f'model initialized with {count_parameters(self.model)} trainable parameters')
        self.log(f'window with step of {self.model.window.w_step} and width {self.model.window.w}')

        # Info image size used in the model
        if self.unsupervised:
            assert self.model.sprites.per_character == 1
            self.best_reco_loss_train = float('inf')
        else:
            self.best_cer_loss_val = float('inf')

    @property
    def best_metric(self):
        if self.unsupervised:
            return self.best_reco_loss_train
        else:
            return self.best_cer_loss_val

    def load_from_dir(self, model_path=None, best=False, resume=None):
        if model_path is None:
            model_dir = self.run_dir
            if best:
                model_path = model_dir / BEST_MODEL
            else:
                model_path = model_dir / MODEL_FILE
        else:
            if best == True:
                import warnings; warnings.warn('Model path explicitly given ignoring "best" argument...')
            model_path = Path(model_path)

        if resume is None:
            resume = self.cfg['training'].get('resume', False)

        msg = f'Load model from path {model_path}'
        self.log(msg)

        assert model_path.exists(), print(f'Couldn\'t find {model_path}')

        checkpoint = torch.load(model_path, map_location=self.device)
        self.start_epoch = 1

        ########### CORE ##########
        self.unsupervised = not checkpoint['supervised']
        if self.unsupervised:
            self.transcribe = None
            self.transcribe_unsupervised = checkpoint["transcribe"]
        else:
            self.transcribe = checkpoint["transcribe"]
        self.transcribe_dataset = checkpoint["transcribe_dataset"]
        self.__post_load__()

        ignore_pretrained = self.cfg['model'].get('ignore_pretrained')
        if ignore_pretrained is not None:
            checkpoint["model_state"] = {k: v for k, v in checkpoint["model_state"].items() if all(not k.startswith(start) for start in ignore_pretrained)}

        keep_pretrained = self.cfg['model'].get('keep_pretrained')
        if keep_pretrained is not None:
            checkpoint["model_state"] = {k: v for k, v in checkpoint["model_state"].items() if any(k.startswith(start) for start in keep_pretrained)}

        self.model.load_state_dict(checkpoint["model_state"])
        ###########################

        if resume:
            self.start_epoch = checkpoint["epoch"]
            if 'best_metric' in checkpoint:
                if self.unsupervised:
                    self.best_reco_loss_train = checkpoint["best_metric"]
                else:
                    self.best_cer_loss_val = checkpoint["best_metric"]
            # ideally we should also compute for the best metric if not already 

            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epoch = self.start_epoch
            self.batch = checkpoint.get("batch", 0)
            self.train_end = False

        if hasattr(self.model, 'cur_epoch'):
            self.model.cur_epoch = checkpoint['epoch']
            self.model.set_back_options()

        self.log(f"checkpoint loaded at epoch {self.start_epoch}")

    def single_train_batch_run(self, x):
        start_time = time()
        B = x['x'].size(0)

        self.model.train()
        self.optimizer.zero_grad()
        y = self.model(x)
        y['loss'].backward()
        self.optimizer.step()

        y['loss'] = y['loss'].item()
        metrics = {'time/img': (time() - start_time) / B}
        for k in y.keys():
            if k.endswith('loss'):
                metrics[k] = y[k] 

        self.train_metrics[x['alias']].update(metrics)

    def step(self):
        self.epoch += 1  # The cur_epoch is starting from 0 whereas in the trainer it is starting from 1
        self.model.step()

    @torch.no_grad()
    def eval_reco(self, split='val'):
        self.model.eval()
        if self.eval:
            self.__post_on_log__() 

        tag = '_' + split
        if split == 'train':
            tag = ''
            assert self.eval or self.train_end, 'Only allowed in eval mode or when train_end is True.'

        for loader in getattr(self, f'{split}_loader'):
            alias = loader.dataset.alias
            for data in loader:
                y = self.model(data)
                y['loss'] = y['loss'].item()

                metrics = {}
                for k in y.keys():
                    if k.endswith('loss'):
                        metrics[k + tag] = y[k]

                getattr(self, f'{split}_metrics')[alias].update(metrics)

    ##### CACHE IMPORTANT THINGS LIKE MATCHING ####
    def save(self, epoch):
        state = {
            "epoch": epoch,
            "model_cfg": self.cfg['model'],
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "supervised": self.supervised,
            "transcribe": (self.transcribe_unsupervised if (self.unsupervised and 'transcribe_unsupervised' in self.__dict__) else self.transcribe),
            "transcribe_dataset": self.transcribe_dataset,
        }
 
        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.log("Model saved at {}".format(save_path))

        if self.save_best_model:
            save = False
            if self.unsupervised:
                reco_loss_train = self.reco_loss_train
                if self.best_reco_loss_train > reco_loss_train:
                    self.log(f'New best model reco-train:{self.best_reco_loss_train} -> {reco_loss_train}')
                    save, self.best_reco_loss_train = True, reco_loss_train
            else:
                cer_loss_val = self.cer_loss_val
                if self.best_cer_loss_val > cer_loss_val:
                    self.log(f'New best model cer-val:{self.best_cer_loss_val} -> {cer_loss_val}')
                    save, self.best_cer_loss_val = True, cer_loss_val

            if save:
                save_path_best_train = join(self.run_dir, BEST_MODEL)
                copyfile(save_path, save_path_best_train)
                self.log(f"saving best model at {save_path_best_train}")
                self.log_images('best')
