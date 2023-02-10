import numpy as np
import torch
import omegaconf
from torch import nn, stack, randn, from_numpy, full, zeros
from scipy import signal

def create_mlp_with_conv1d(in_ch, out_ch, n_hidden_units, n_layers, norm_layer=None):
    if norm_layer is None or norm_layer in ['id', 'identity']:
        norm_layer = nn.Identity
    elif norm_layer in ['batch_norm', 'bn']:
        norm_layer = nn.BatchNorm1d
    elif not norm_layer == nn.BatchNorm1d:
        raise NotImplementedError

    if n_layers > 0:
        seq = [nn.Conv1d(in_ch, n_hidden_units, kernel_size=1), norm_layer(n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Conv1d(n_hidden_units, n_hidden_units, kernel_size=1), nn.ReLU(True)]
        seq += [nn.Conv1d(n_hidden_units, out_ch, kernel_size=1)]
    else:
        seq = [nn.Conv1d(in_ch, out_ch, kernel_size=1)]
    return nn.Sequential(*seq)

def create_gaussian_weights(img_size, n_channels, std=6):
    g1d_h = signal.gaussian(img_size[0], std)
    g1d_w = signal.gaussian(img_size[1], std)
    g2d = np.outer(g1d_h, g1d_w)
    return from_numpy(g2d).unsqueeze(0).repeat(n_channels, 1, 1).float()

def init_objects(K, c, size, init):
    samples = []
    for _ in range(K):
        if 'constant' in init:
            cons = init['constant']
            if isinstance(cons, omegaconf.listconfig.ListConfig):
                sample = zeros((c, ) + tuple(size), dtype=torch.float)
                for c_id in range(c):
                    sample[c_id, ...] = cons[c_id]
                # sample[]
            else:
                sample = full((c, ) + tuple(size), cons, dtype=torch.float)
        elif 'gaussian' in init:
            sample = create_gaussian_weights(size, c, init['gaussian'])
        else:
            raise NotImplementedError
        samples.append(sample)

    return stack(samples)

def copy_with_noise(t, noise_scale=0.0001):
    return t.detach().clone() + randn(t.shape, device=t.device) * noise_scale
