from itertools import chain
from collections import Counter
import torch
import numpy as np

from learnable_typewriter.evaluate.quantitative.sprite_matching.metrics import error_rate 
from learnable_typewriter.evaluate.quantitative.sprite_matching.data import Data, Dataset
from learnable_typewriter.evaluate.quantitative.sprite_matching.trainer import TrainerBatch, Trainer

def metrics_to_average_sub(v):
    return {p: (q if p not in {'cer', 'wer', 'ser'} else np.mean(q)) for p, q in v.items()}

def metrics_to_average(obj):
    return {k: metrics_to_average_sub(v) for k, v in obj.items()}

def optimize_dict(trainer, mapping):
    loader = trainer.get_dataloader(split='train', percentage=0.1, batch_size=4, num_workers=trainer.n_workers, remove_crop=True)[0]
    data = Data(trainer, loader)
    map_gt = dict(trainer.transcribe_dataset)
    # needs to be fixed to find character that doesn't exist in the dataset
    map_gt[-1] =  ('@' if '_' == loader.dataset.dataset.space else '_') 

    stats = Counter(a for c in data for b in c for a in b)
    def key(k):
        return stats.get(k, 0)

    sep, space = loader.dataset.dataset.sep, loader.dataset.dataset.space
    base_max = error_rate(data, average=True, delim=space, sep=sep, map_pd=mapping, map_gt=map_gt)['cer']
    for i in sorted(range(len(trainer.model.sprites)), key=key):
        temp, mapping[i] = mapping[i], space
        new_error = error_rate(data, average=True, delim=space, sep=sep, map_pd=mapping, map_gt=map_gt)['cer']
        if new_error > base_max:
            mapping[i] = temp
        else:
            base_max = new_error

    return mapping

def er_evaluate_unsupervised(
        trainer,
        mapping=None,
        matching_model_batch_size=256,
        lr=1,
        dataloader_batch_size=4,
        train_percentage=0.1,
        batch=True,
        cer_num_epochs=10,
        verbose=True,
        average=True,
        optimize=True,
    ):
    trainer.model.eval()
    trainer.log(f'Unsupervised Evaluation on {train_percentage}% of the data with a batch size of {trainer.batch_size}', eval=True)

    # if mapping is None:
    trainer.log('Inferring mapping', eval=True)
    train_loader = trainer.get_dataloader(split='train', percentage=train_percentage, batch_size=dataloader_batch_size, num_workers=trainer.n_workers, remove_crop=True)
    TrainerClass = (TrainerBatch if batch else Trainer)
    data_train = Data(trainer, chain(*train_loader), tag=('er-build-dataset-train' if verbose else None))
    dataset = Dataset(data_train, A=len(trainer.transcribe_dataset), S=len(trainer.model.sprites))
    with torch.enable_grad():
        tr = TrainerClass(dataset, device=trainer.device, verbose=verbose)
        tr.train(num_epochs=cer_num_epochs, lr=lr, batch_size=matching_model_batch_size, print_progress=verbose)
    mapping = {s: trainer.transcribe_dataset[a] for s, a in tr.mapping.items()}
    trainer.log(f'Inferred mapping: {mapping}', eval=True)

    mapping_ = dict(mapping)
    if optimize:
        mapping_ = optimize_dict(trainer, mapping_)

    output = {}
    loaders_ = {
        'train_loader': trainer.get_dataloader(split='train', batch_size=dataloader_batch_size, num_workers=trainer.n_workers, shuffle=False, remove_crop=True),
        'test_loader': trainer.get_dataloader(split='test', batch_size=dataloader_batch_size, num_workers=trainer.n_workers, shuffle=False, remove_crop=True)
    }

    map_gt = dict(trainer.transcribe_dataset)
    for split, loaders in loaders_.items():
        for loader in loaders:
            map_gt[-1] =  ('@' if '_' == loader.dataset.space else '_')
            output[(loader.dataset.alias, split.split('_')[0])] = error_rate(Data(trainer, loader), verbose=verbose, average=average, delim=loader.dataset.space, sep=loader.dataset.sep, map_pd=mapping_, map_gt=map_gt)

    output = {'metrics': output, 'mapping': mapping}
    output.update(loaders_) 
    return output

def er_evaluate_supervised(trainer, verbose=False, average=True, eval_train=True, dataloader_batch_size=None, splits=None):
    trainer.model.eval()
    
    loaders_ = {'val': trainer.val_loader, 'test': trainer.test_loader}
    if eval_train:
        loaders_['train'] = trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True, remove_crop=True)

    output = {}
    map_pd, map_gt = trainer.transcribe, dict(trainer.transcribe)
    for split, loaders in loaders_.items():
        if splits is not None and split not in splits:
            continue

        for loader in loaders:
            map_gt[-1] =  ('@' if '_' == loader.dataset.space else '_')
            labels = []
            for x in loader:
                labels += list(zip(trainer.inference(x), x['y']))

            output[(loader.dataset.alias, split)] = error_rate(labels, verbose=verbose, average=average, delim=loader.dataset.space, sep=loader.dataset.sep, map_pd=map_pd, map_gt=map_gt)

    output = {'metrics': output, 'mapping': trainer.transcribe}
    output.update(loaders_) 
    return output
