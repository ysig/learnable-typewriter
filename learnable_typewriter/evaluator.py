from itertools import chain
from collections import defaultdict

import pandas as pd
import numpy as np
import torch

from learnable_typewriter.evaluate.quantitative.metrics import Metrics, AverageMeter
from learnable_typewriter.evaluate.quantitative.sprite_matching.evaluate import er_evaluate_unsupervised, er_evaluate_supervised, metrics_to_average_sub
from learnable_typewriter.evaluate.qualitative.decompositor import Decompositor
from learnable_typewriter.logger import Logger


class Evaluator(Logger):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""
    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.post_load_flag:
            self.__post_init_evaluate__()

    def __post_init_evaluate__(self):
        self.__init_decompositor__()
        self.__init_er__()
        self.__init_metrics__()

    def __init_decompositor__(self):
        self.decompositor = Decompositor(self)
        self.eval_qualitative = self.log_cfg.get("qualitative", False)

    def __init_er__(self):
        self.compute_er_mask = self.has_labels
        self.compute_er_last = self.log_cfg['val']['error_rate'].get('last', True)
        self.eval_best = self.log_cfg['val']['error_rate'].get('eval_best', False)
        self.eval_best_str = '_best'
        self.eval_best_mode = False
        self.error_rate_kargs = self.log_cfg['val']['error_rate'].get('kargs', {})

        if self.compute_er_mask:
            if not self.eval:
                columns = ["iteration", 'cer', 'ser']
                self.df_er = defaultdict(lambda : pd.DataFrame(columns=columns))

    def __init_metrics__(self):
        # Train metrics
        metric_names = ['time/img', 'loss', 'reco_loss']
        self.train_metrics = {ld.dataset.alias: Metrics(*metric_names, sum_for_prop=True) for ld in self.train_loader}
        self.test_metrics = {ld.dataset.alias: Metrics('loss_test', 'reco_loss_test') for ld in self.test_loader}
        self.metrics_ = {'train': self.train_metrics, 'test': self.test_metrics}

        if self.supervised:
            self.val_metrics = {ld.dataset.alias: Metrics('loss_val', 'reco_loss_val') for ld in self.val_loader}
            self.metrics_['val'] = self.val_metrics

    def log_er(self, average):
        # Print & write metrics to file update existing data
        self.log(f'computing error-rate')
        self.cer_loss_val_ = []
        tid = ('best-model:' if self.eval_best_mode else '')
        for (alias, split), metrics in average.items():
            for k, v in metrics.items():
                if k in {'cer', 'wer', 'ser'}:
                    self.tensorboard.add_scalar(f'metrics/{alias}/{k}/{split}/', v, self.cur_iter)
                    self.log(f'{tid}[{alias}/{split}] {k}:{v}', eval=True)
                    if k == 'cer' and split == 'val':
                        self.cer_loss_val_.append(v)

    def log_er_texts(self, metadata, mapping, max_lines=10):
        if self.eval:
            return

        for (alias, split), metrics in metadata.items():
            pred, gt = metrics['texts'], metrics['gt']
            pred, gt = pred[:max_lines], gt[:max_lines]
            for i, (p, g) in enumerate(zip(pred, gt)):
                if self.eval_best_mode:
                    self.tensorboard.add_text(f'{alias}/{split}/{i}/pred{self.eval_best_str}', f'`{p}`', self.cur_iter)
                else:
                    self.tensorboard.add_text(f'{alias}/{split}/{i}/pred+gt', '  \n'.join([f'`{p}`', f'`{g}`']), self.cur_iter)
        self.tensorboard.add_text(f'{alias}/{split}/mapping', str(mapping), self.cur_iter)

    def log_er_last(self, average):
        self.log_er(average)

        metric_dict = {}
        for (alias, split), metrics in average.items():
            for er, v  in metrics.items():
                if er in {'cer', 'wer', 'ser'}:
                    k = f'metrics/{er}/{alias}/{split}'
                    metric_dict.update({k: v})
                    self.log(f'{k}: {v}', eval=True)

        if not self.eval:
            self.tensorboard.add_hparams(self.cfg_flat, metric_dict)

    @property
    def cer_loss_val(self):
        if 'cer_loss_val_' in self.__dict__:
            return np.mean(self.cer_loss_val_)
        else:
            kargs = dict(*self.error_rate_karg)
            kargs['splits'], kargs['average'] = ['val'], True
            error_rates = er_evaluate_supervised(self, **self.error_rate_karg)['metrics']
            return np.mean([v['cer'] for e, v in error_rates.items()])

    def er_evaluate_(self, average):
        if self.transcribe is None:
            return er_evaluate_unsupervised(self, average=average, **self.error_rate_kargs, **({'mapping': self.transcribe_unsupervised} if self.eval else {}))
        else:
            return er_evaluate_supervised(self, average=average, **self.error_rate_kargs)

    @torch.no_grad()
    def error_rate(self, eval_best=False):
        if self.eval:
            self.__post_on_log__()

        if self.eval or self.train_end:
            if eval_best:
                self.__resume__(best=eval_best)
                self.eval_best_mode = True

            self.last_error_rate()

            if eval_best:
                self.eval_best_mode = False
        else:
            self.log(f'evaluating error-rates')
            er = self.er_evaluate_(average=False)
            average = {k: metrics_to_average_sub(v) for k, v in er['metrics'].items()}
            self.log_er(average)
            if self.transcribe is None:
                self.transcribe_unsupervised = er['mapping']
            self.log_er_texts(er['metrics'], mapping=er['mapping'])

    def last_error_rate(self):
        er = self.er_evaluate_(average=False)
        average = {k: metrics_to_average_sub(v) for k, v in er['metrics'].items()}
        if self.transcribe is None:
            self.transcribe_unsupervised = er['mapping']
        self.log_er_texts(er['metrics'], mapping=er['mapping'])
        self.log_er_last(average)

    @torch.no_grad()
    def recons_loss_eval(self, set_="train"):
        """Routine to save quantitative results for sequence predictions: loss + scores on validation set"""
        assert set_ in {'train', 'val'}
        self.model.eval()

        loss = AverageMeter()
        loader = getattr(self, f'{set_}_loader')
        for data in chain(*loader):
            y = self.model(data['x'], return_recons_loss=True)
            loss.update(y['recons_loss'], n=len(data['x']))

        self.log(f"Reconstruction loss for {set_} set: {loss.avg:.4f}")
        if not self.eval:
            scores_path = self.run_dir / f'final_recons_loss_{set_}.tsv'
            with open(scores_path, mode="w") as f:
                f.write("recons_loss\t" + "\n")
                f.write("{:.6}\t".format(loss.avg) + "\n")
