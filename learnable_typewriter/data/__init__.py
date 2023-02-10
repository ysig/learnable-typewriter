from learnable_typewriter.data.dataset import LineDataset

def get_dataset(line=True, **kargs):
    dataset_args = dict(kargs)
    if line:
        del dataset_args['crop_width']
    return LineDataset(**dataset_args)
