from torch import nn

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def create_mlp(in_ch, out_ch, n_hidden_units, n_layers, norm_layer=None):
    if norm_layer is None or norm_layer in ['id', 'identity']:
        norm_layer = Identity
    elif norm_layer in ['batch_norm', 'bn']:
        norm_layer = nn.BatchNorm1d
    elif not norm_layer == nn.BatchNorm1d:
        raise NotImplementedError

    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_hidden_units), norm_layer(n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(True)]
        seq += [nn.Linear(n_hidden_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    return nn.Sequential(*seq)

def create_mlp_with_conv1d(in_ch, out_ch, n_hidden_units, n_layers, norm_layer=None):
    if norm_layer is None or norm_layer in ['id', 'identity']:
        norm_layer = Identity
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

def get_nb_out_channels(layer):
    return list(filter(lambda e: isinstance(e, nn.Conv2d), layer.modules()))[-1].out_channels
