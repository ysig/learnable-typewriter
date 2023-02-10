from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from learnable_typewriter.typewriter.typewriter.sprites.utils import create_mlp_with_conv1d, copy_with_noise

########################
#  ABC
########################

class _AbstractTransformationModule(nn.Module):
    __metaclass__ = ABCMeta
    identity_module = False

    @abstractmethod
    def transform(self, x, beta):
        return self._transform(x, beta)

    def __bool__(self):
        return not self.identity_module

    def load_with_noise(self, module, noise_scale):
        if bool(self):
            self.load_state_dict(module.state_dict())
            self.mlp[-1].bias.data.copy_(copy_with_noise(module.mlp[-1].bias, noise_scale))

    @property
    def dim_parameters(self):
        try:
            dim_parameters = self.mlp[-1].out_features
        except AttributeError as e:
            dim_parameters = self.mlp[-1].out_channels
        return dim_parameters


########################
#   Modules 
########################

class IdentityModule(_AbstractTransformationModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.identity_module = True
 
    def predict_parameters(self, x, *args, **kargs):
        return x

    def transform(self, x, *args, **kwargs):
        return x

    def load_with_noise(self, module, noise_scale):
        pass


class ColorModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.color_ch = kwargs['color_channels']
        n_layers = kwargs['n_hidden_layers']
        n_hidden_units = kwargs['n_hidden_units']

        # MLP
        self.mlp = create_mlp_with_conv1d(in_channels, self.color_ch, n_hidden_units, n_layers)
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()

        # Identity transformation parameters
        self.register_buffer('identity', torch.eye(self.color_ch, self.color_ch))

    def predict_parameters(self, x):
        return self.mlp(x)

    def transform(self, x, beta):
        if x.size(1) == 2 or x.size(1) > 3:
            x, mask = torch.split(x, [self.color_ch, x.size(1) - self.color_ch], dim=1)
        else:
            mask = None

        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)

        weight = beta.view(-1, self.color_ch, 1)
        weight = weight.expand(-1, -1, self.color_ch) * self.identity + self.identity

        output = torch.einsum('bij, bjkl -> bikl', weight, x)
        output = torch.sigmoid(output)
        output = torch.clamp(output, 0, 1)
        if mask is not None:
            output = torch.cat([output, mask], dim=1)
        
        return output

class PositionModule(_AbstractTransformationModule):
    def __init__(self, in_channels, canvas_size, **kwargs):
        super().__init__()
        self.padding_mode = kwargs['padding_mode']
        self.parametrization = kwargs['parametrization']

        if self.parametrization not in ['exp', 'sinh']:
            raise ValueError(self.parametrization)

        self.Hs, self.Ws = kwargs['sprite_size']
        self.H, self.W = int(canvas_size[0]), int(canvas_size[1])

        # MLP Init
        n_layers = kwargs['n_hidden_layers']
        n_hidden_units = kwargs['n_hidden_units']

        self.mlp = create_mlp_with_conv1d(in_channels, 3, n_hidden_units, n_layers)
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()

        # Spatial constraint
        self.register_buffer('t', torch.Tensor([kwargs['max_x'], kwargs['max_y']]).unsqueeze(0).unsqueeze(-1))

        # Identity transformation parameters
        eye = torch.eye(2, 2)
        eye[0, 0], eye[1, 1] = self.W/self.Ws, self.H/self.Hs
        self.register_buffer('eye', eye)

    def predict_parameters(self, x):
        beta = self.mlp(x)

        s, t = beta.split([1, 2], dim=1)

        if self.parametrization == 'exp':
            s = torch.exp(s)
        elif self.parametrization == 'sinh':
            s = torch.sinh(s)

        t = torch.clamp(t, min=-1.0, max=1.0)*self.t

        return torch.cat([s, t], dim=1)

    def transform(self, x, beta):
        s, t = beta.split([1, 2], dim=1)
        scale = s[..., None].expand(-1, 2, 2) * self.eye
        beta = torch.cat([scale, t.unsqueeze(2)], dim=2)
 
        # grid is a batch of affine matrices
        grid = F.affine_grid(beta, (x.size(0), x.size(1), self.H, self.W), align_corners=False)

        # The size-2 vector grid[n, h, w] specifies input pixel locations x and y with n = batch size * nb_sprites + empty sprite
        return F.grid_sample(x, grid, mode='bilinear', padding_mode=self.padding_mode, align_corners=False)
