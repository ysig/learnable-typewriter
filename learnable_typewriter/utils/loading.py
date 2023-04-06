import sys
from pathlib import Path
from omegaconf import OmegaConf

from learnable_typewriter.trainer import Trainer


def load_pretrained_model(path, device='cpu', default_run_dir=None, default_dataset_path=None, kargs=None, drop_crop=True):
    """ dataset = dataset alias """
    run_path = Path(path)
    try:
        config_path = next(run_path.glob('*.yaml'))
    except StopIteration:
        raise StopIteration

    cfg = OmegaConf.load(config_path)
    cfg['eval'] = True
    training = cfg["training"]
    training['device'] = device

    if drop_crop:
        for k in cfg['dataset'].keys():
            if 'crop_width' in cfg['dataset'][k]:
                del cfg['dataset'][k]['crop_width']

    if kargs is not None:
        cache = list(sys.argv)
        sys.argv = [__file__] + kargs.split(' ')
        cfg.merge_with_cli()
        sys.argv = cache

    cfg['run_dir'] = path

    if default_run_dir is not None:
        cfg['default_run_dir'] = default_run_dir

    if default_dataset_path is not None:
        cfg['default_dataset_path'] = default_dataset_path

    trainer = Trainer(cfg)
    trainer.run_init()
    trainer.model.eval()
    return trainer