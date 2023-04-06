import numpy as np
import torch
from torch import nn, ones
from learnable_typewriter.typewriter.utils import get_clamp_func
from learnable_typewriter.typewriter.typewriter.sprites.utils import init_objects
from learnable_typewriter.typewriter.typewriter.sprites.unet import UNet


class Generator(nn.Module):
    def __init__(self, n_outputs, sprite_size, logger, type='mlp'):
        super().__init__()
        self.mode = type
        self.proto = nn.Parameter(torch.rand((n_outputs, 1) + tuple(sprite_size)))  #size (K,1,H,W)
        self.flat_latents = self.flat_latents_
        self.latent_dim = sprite_size[0]*sprite_size[1]

        if type == 'mlp':
            logger('Generator is MLP')
            self.gen = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            )
        elif type == 'marionette':
            logger('Generator is Marionette')
            self.latent_dim = 128
            assert self.latent_dim*8 == sprite_size[0]*sprite_size[1], '`marionette` mode is for evaluation only working with 32*32 sprites'
            del self.proto

            self.output_size = (n_outputs, 1, sprite_size[0], sprite_size[1])
            self.proto = nn.Parameter(torch.rand((n_outputs, 128)))
            self.gen = nn.Sequential(
                nn.Linear(self.latent_dim, 8*self.latent_dim),
                nn.GroupNorm(8, 8*self.latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(8*self.latent_dim, sprite_size[0]*sprite_size[1]),
                nn.Sigmoid()
            )

            self.flat_latents = self.flat_latents_marionette_
        else:
            logger('Generator is UNET')
            self.gen = UNet(1, 1)

        self.forward = (self.forward_mlp if type=='mlp' else (self.forward_marionette if type == 'marionette' else self.forward_unet))
        self.forward()

    def forward_mlp(self, x=None):
        size = self.proto.size()
        return self.gen(self.proto.flatten(start_dim=-3, end_dim=-1)).reshape(*size) #size (K,1,H,W)

    def forward_marionette(self, x=None):
        return self.gen(self.proto).reshape(*self.output_size) #size (K,1,H,W)

    def forward_unet(self, x=None):
        return torch.sigmoid(self.gen(self.proto)) #size (K,1,H,W)

    def flat_latents_(self):
        return self.proto.squeeze(1).flatten(start_dim=-2)

    def flat_latents_marionette_(self):
        return self.proto


def min_max(masks):
    masks = masks.detach()
    masks -= masks.flatten(start_dim=1).min(1, keepdim=True)[0].unsqueeze(1).unsqueeze(1)
    masks /= masks.flatten(start_dim=1).max(1, keepdim=True)[0].unsqueeze(1).unsqueeze(1)
    masks[masks < 0.4] = 0
    return masks

def idem(x):
    return x

class Sprites(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()

        # Prototypes & masks
        self.n_sprites = cfg['n']
        self.per_character = cfg.get('L', 1)
        self.sprite_size = cfg['size']
        self.proto_init = cfg['init']['color']
        self.color_channels = cfg['color_channels']
        samples = init_objects(self.n_sprites*self.per_character, self.color_channels, self.sprite_size, self.proto_init)
        self.prototypes = nn.Parameter(samples)
        self.prototypes.requires_grad = False

        self.masks_ = Generator(self.n_sprites*self.per_character, self.sprite_size, type=cfg['gen_type'], logger=logger)
        self.frozen = False

        self.active_prototypes = ones(len(self))
        self.clamp_func = get_clamp_func(cfg['use_clamp'])

    def __len__(self):
        return self.n_sprites*self.per_character

    @property
    def masks(self):
        masks = self.masks_()
        return masks if self.training else self.clamp_func(masks)

