"""Stage 1"""
import torch
from itertools import chain

from learnable_typewriter.utils.generic import use_seed, alternate
from learnable_typewriter.typewriter.optim.optimizer import get_optimizer
from learnable_typewriter.utils.milestone import Milestone
from learnable_typewriter.evaluator import Evaluator
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)


class Trainer(Evaluator):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""
    @use_seed()
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.post_load_flag:
            self.__resume__()
        else:
            self.__post_init_trainer__()

    def __post_init_trainer__(self):
        self.__init_training__()
        self.__init_optimizer__()
        self.__init_milestone__()
        self.cache_config()

    def __post_on_log__(self):
        if not self.__post_on_log_flag__:
            self.__post_on_log_flag__ = True
            self.__post_init_logger__()
            self.__post_init_evaluate__()

    def __post_load__(self):
        self.__post_init_dataset__()
        self.__post_init_model__()
        self.__post_on_log_flag__ = False
        self.__init_decompositor__()
        if not self.eval:
            self.__post_on_log__()
        else:
            self.__init_metrics__()
        self.__post_init_trainer__()

    def __init_training__(self):
        self.n_batches = sum(len(dl) for dl in self.train_loader)
        self.n_epochs = self.cfg["training"]["num_epochs"]
        self.flush_memory = self.cfg['training']['flush_mem']
        self.flush_period = self.cfg['training']['flush_per']

    def __init_optimizer__(self):
        # Optimizer
        opt_params = dict(self.cfg["training"]["optimizer"]) or {}
        optimizer_name = opt_params.pop("name")
        prototypes_kwargs = opt_params.pop('prototypes', {})
        tsf_kwargs = opt_params.pop('transformation', {})
        encoder_kwargs = opt_params.pop('encoder', {})

        self.optimizer = get_optimizer(optimizer_name)([
            dict(params=self.model.prototypes_parameters(), **prototypes_kwargs),
            dict(params=self.model.transformation_parameters(), **tsf_kwargs),
            dict(params=chain(self.model.encoder.parameters(), self.model.selection.sprite_params(), self.model.selection.encoder_params()), **encoder_kwargs)],
            **opt_params)
        self.model.set_optimizer(self.optimizer)

        self.log("Optimizer:\n" + str(self.optimizer))

    def __init_milestone__(self):
        events = {}
        events['train.reconstruction'] = self.log_cfg['train']['reconstruction'].get('every', 0)
        events['train.images'] = self.log_cfg['train']['images'].get('every', 0)
        events['val.reconstruction'] = self.log_cfg['val']['reconstruction'].get('every', 0)
        events['val.error_rate'] = self.log_cfg['val']['error_rate'].get('every', 0)
        events['save'] = self.log_cfg['save'].get('every', 0)
        self.save_best_model = self.log_cfg["save"]["best"]
        self.milestone = Milestone(self.log_cfg['milestone'], events=events)

    def __resume__(self, best=False):
        checkpoint_path_resume = self.cfg["training"].get("pretrained")
        if checkpoint_path_resume is not None:
            print('Load previous checkpoint:', checkpoint_path_resume)
        self.load_from_dir(checkpoint_path_resume, best=best)

    def do_milestone(self):
        self.milestone.update()
        flags = self.milestone.get()
        self.compute_metrics(flags=flags)

    def run_step(self, x):
        self.single_train_batch_run(x)

    def save_model(self):
        self.save(epoch=self.epoch)

    @torch.no_grad()
    def compute_metrics(self, msg=None, flags={}):
        if msg is not None:
            self.log(msg)

        if flags.get('train.reconstruction', self.train_end) and not self.evaluate_only:
            self.log_train_metrics()

        val_flag = flags.get('val.reconstruction', self.train_end) and self.val_flag
        if val_flag:
            self.log_val_metrics()

        if flags.get('train.images', self.train_end):
            self.log_images()

        if flags.get('val.error_rate', self.train_end and self.compute_er_last) and self.compute_er_mask:
            self.error_rate()
            if self.train_end and self.eval_best:
                self.error_rate(eval_best=True)

        if (flags.get('save', self.train_end) or self.train_end) and not self.evaluate_only:
            self.save_model()

        self.reset_metrics()

    def run_init(self):
        self.train_end = False
        self.evaluate_only = False
        self.reset_val, self.reset_train = False, False

        self.batch = 1
        self.epoch = getattr(self, 'start_epoch', 1)
        self.cur_iter = (self.epoch - 1) * self.n_batches - 1
        self.prev_train_step, self.prev_val_step = self.cur_iter, self.cur_iter

    @use_seed()
    def run(self):
        self.run_init()
        if self.eval:
            message = "Evaluating"
            self.evaluate_only = True
            self.__post_init_logger__()
            self.__post_init_evaluate__()

        else:
            if self.epoch > 1:
                message = "Compute Validation losses"
                flags = {'val.reconstruction': True, 'train.images': True}
                self.compute_metrics(msg=message, flags=flags)

            self.log('training starts')
            torch.cuda.empty_cache()

            for self.epoch in range(self.epoch, self.n_epochs + 1):
                for self.batch, batch_data in enumerate(alternate(*self.train_loader), start=1):
                    self.cur_iter += batch_data['x'].size()[0]
                    self.run_step(batch_data)

                    del batch_data
                    if self.batch%self.flush_period == 0 and self.flush_memory:
                        torch.cuda.empty_cache()

                self.do_milestone()
                self.step()

            message = "Training finished - evaluating"

        self.train_end = True
        self.call_it_a_day()
        self.compute_metrics(msg=message)
        if 'tensorboard' in self.__dict__ and self.tensorboard is not None:
            self.__close_tensorboard__()
        self.log('Finished.')

    def call_it_a_day(self):
        self.epoch = self.n_epochs
        self.batch = len(self.train_loader)
